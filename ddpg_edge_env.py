import gymnasium
import numpy as np
from EdgeSimPy.core.simulator import Simulator
from EdgeSimPy.components.edge_server import EdgeServer
from EdgeSimPy.components.network_link import NetworkLink
from EdgeSimPy.components.base_station import BaseStation
from EdgeSimPy.components.network_switch import NetworkSwitch
from EdgeSimPy.components.power_models.servers.linear import LinearPowerModel
from EdgeSimPy.components.service import Service
from EdgeSimPy.components.application import Application
from EdgeSimPy.components.container_image import ContainerImage
# Import other components that might be registered with the simulator
from EdgeSimPy.components.user import User
from EdgeSimPy.components.container_image import ContainerLayer, ContainerRegistry # Keep if needed, ContainerImage is main one
from EdgeSimPy.components.network_flow import NetworkFlow
from EdgeSimPy.components.topology import Topology


class TaskOffloadingEnv(gymnasium.Env):
    def __init__(self, config=None):
        super().__init__()

        # Initialize Simulator
        # Ensure resource_management_algorithm is None as DDPG agent handles decisions
        self.simulator = Simulator(
            tick_duration=1, 
            tick_unit="milliseconds",
            resource_management_algorithm=None 
        )

        # Define the dataset for EdgeSimPy initialization
        dataset = {
            "EdgeServer": [
                {
                    "attributes": {"id": 1, "model_name": "EdgeDevice", "cpu": 1000, "memory": 4096, "disk": 102400},
                    "relationships": {"power_model": "LinearPowerModel", "base_station": {"class": "BaseStation", "id": 1}}
                },
                {
                    "attributes": {"id": 2, "model_name": "FogNode", "cpu": 4000, "memory": 16384, "disk": 512000},
                    "relationships": {"power_model": "LinearPowerModel", "base_station": {"class": "BaseStation", "id": 2}}
                }
            ],
            "BaseStation": [
                {"attributes": {"id": 1}, "relationships": {"network_switch": {"class": "NetworkSwitch", "id": 1}}},
                {"attributes": {"id": 2}, "relationships": {"network_switch": {"class": "NetworkSwitch", "id": 1}}}
            ],
            "NetworkSwitch": [
                {"attributes": {"id": 1}, "relationships": {}}
            ],
            "NetworkLink": [
                {
                    "attributes": {"id": 1, "bandwidth": 100, "delay": 5},
                    "relationships": {"nodes": [{"class": "BaseStation", "id": 1}, {"class": "BaseStation", "id": 2}]}
                }
            ],
            "ContainerImage": [
                {"attributes": {"digest": "dummy_image_digest", "name": "dummy_image", "layers_digests": []}, "relationships": {}}
            ],
            "Application": [
                {"attributes": {"id": 1, "label": "default_app"}, "relationships": {}}
            ]
        }

        # Add empty lists for other known EdgeSimPy components not explicitly defined above
        # Service will be created dynamically. User, ContainerLayer, ContainerRegistry, NetworkFlow, Topology might not be needed for this specific env.
        component_names = ["User", "ContainerLayer", "ContainerRegistry", "NetworkFlow", "Topology", "Service"]
        for comp_name in component_names:
            if comp_name not in dataset:
                dataset[comp_name] = []

        # Initialize the simulation environment
        self.simulator.initialize(dataset=dataset)

        # Manually set Power Models after initialization
        for server in EdgeServer.all():
            if server.id == 1:  # EdgeDevice
                server.power_model = LinearPowerModel(power_per_unit_load=0.5, idle_power_consumption=10)
            elif server.id == 2:  # FogNode
                server.power_model = LinearPowerModel(power_per_unit_load=0.7, idle_power_consumption=50)

        # Define action space (normalized decision: -1 for local, 1 for fog)
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define observation space (normalized values)
        self.observation_space = gymnasium.spaces.Dict({
            "edge_cpu_load": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "fog_cpu_load": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "task_cpu_demand": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "network_latency_to_fog": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        # Store references to servers
        self.edge_device = EdgeServer.find_by_id(1)
        self.fog_node = EdgeServer.find_by_id(2)

        # Task-related attributes
        self.task_counter = 0
        self.current_task = None  # This will hold the current Service object
        self.max_task_cpu_demand = 500.0  # MIPS, for normalization
        self.max_network_latency = 10.0  # ms, for normalization (e.g. link delay + processing if any)
        self.application = Application.find_by_id(1) # Store reference to the default application

        # Custom attributes for the environment
        self.processing_duration_ticks = 10
        self.w_latency = 1.0
        self.w_energy = 0.5
        self.max_steps_per_episode = 100
        self.current_step_in_episode = 0


    def _generate_task(self):
        self.task_counter += 1
        task_cpu = np.random.randint(50, int(self.max_task_cpu_demand) + 1)  # Ensure max_task_cpu_demand is int for randint
        
        # Create a new service instance.
        service = Service(
            label=f"task_{self.task_counter}",
            image_digest="dummy_image_digest",
            cpu_demand=task_cpu,
            memory_demand=128  # Can be fixed or randomized
        )
        # Assign the application to the service
        service.application = self.application
        # NOTE: The service is not yet provisioned on any server or added to the simulator schedule.
        # This will happen in the `step` method based on the agent's action.
        return service

    def _get_observation(self):
        edge_cpu_load_abs = self.edge_device.cpu_demand if self.edge_device else 0
        edge_cpu_total = self.edge_device.cpu if self.edge_device else 1 # Avoid division by zero
        edge_load = np.float32(edge_cpu_load_abs / edge_cpu_total) if edge_cpu_total > 0 else np.float32(0)

        fog_cpu_load_abs = self.fog_node.cpu_demand if self.fog_node else 0
        fog_cpu_total = self.fog_node.cpu if self.fog_node else 1 # Avoid division by zero
        fog_load = np.float32(fog_cpu_load_abs / fog_cpu_total) if fog_cpu_total > 0 else np.float32(0)
        
        task_demand_normalized = np.float32(self.current_task.cpu_demand / self.max_task_cpu_demand) \
            if self.current_task and self.max_task_cpu_demand > 0 else np.float32(0)
        
        # Get latency from the link connecting BaseStation 1 (edge) and BaseStation 2 (fog)
        # This assumes link ID 1 connects them, as defined in the dataset.
        link = NetworkLink.find_by_id(1)
        network_lat = np.float32(link.delay / self.max_network_latency) \
            if link and self.max_network_latency > 0 else np.float32(0)

        return {
            "edge_cpu_load": np.clip(edge_load, 0, 1).reshape(1),
            "fog_cpu_load": np.clip(fog_load, 0, 1).reshape(1),
            "task_cpu_demand": np.clip(task_demand_normalized, 0, 1).reshape(1),
            "network_latency_to_fog": np.clip(network_lat, 0, 1).reshape(1)
        }

    def _get_reward(self):
        # Placeholder for reward calculation logic
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset EdgeSimPy simulation state
        services_to_remove = list(Service.all())
        for service in services_to_remove:
            if service.server is not None:
                service.server.cpu_demand -= service.cpu_demand
                service.server.memory_demand -= service.memory_demand
                if service in service.server.services:
                    service.server.services.remove(service)
            
            # Remove from scheduler
            if service in self.simulator.schedule.agents:
                self.simulator.schedule.remove(service)
            
            # Remove from global list of services
            if service in Service._instances:
                 Service._instances.remove(service)
        
        # Reset server demands and states
        for server in EdgeServer.all():
            server.cpu_demand = 0
            server.memory_demand = 0
            server.disk_demand = 0  # Assuming disk might be used
            server.ongoing_migrations = 0 # Reset migrations if tracked

        # Reset simulator's clock/schedule
        self.simulator.schedule.steps = 0
        self.simulator.schedule.time = 0
        self.simulator.last_dump = 0  # Reset last dump time if logging is used

        # Reset task counter for consistent task labels across episodes if desired, or keep incrementing
        self.task_counter = 0 
        self.current_step_in_episode = 0

        # Generate an initial task
        self.current_task = self._generate_task()

        # Calculate the initial observation
        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        # 1. Interpret Action
        decision_threshold = 0.0
        if action[0] < decision_threshold:
            chosen_server = self.edge_device
        else:
            chosen_server = self.fog_node

        # 2. Simulate Task Provisioning and Execution
        task_to_provision = self.current_task
        if not isinstance(task_to_provision, Service):
            raise TypeError(f"current_task is not a Service object, but {type(task_to_provision)}")
        
        task_to_provision.application = self.application # Should be already set by _generate_task

        # Store initial energy for delta calculation if needed, or just use get_power_consumption
        # initial_energy_chosen_server = chosen_server.power_model.energy_consumed
        
        # Provision the task. This places the service on the server and adds it to the simulator's schedule.
        task_to_provision.provision(target_server=chosen_server)

        # Run the simulator for processing_duration_ticks
        # sim_start_time_step = self.simulator.schedule.steps # Not strictly needed for fixed duration
        for _ in range(self.processing_duration_ticks):
            self.simulator.step()
        
        # 3. Calculate Latency
        latency = 0.0
        
        if not task_to_provision._Service__migrations:
            # This case should ideally not happen if provision worked and created a migration record.
            # Handle this potential error, maybe by assigning a high penalty or logging.
            # For now, let's assume migration_info will be available.
            # If it means the task could not be provisioned (e.g. server full),
            # the reward should reflect that (e.g. high penalty).
            # This might require checking if chosen_server actually has the service.
            # For now, proceeding with assumption it's there.
            print(f"Warning: No migration info for task {task_to_provision.label} on {chosen_server.label}")
            provisioning_latency_ticks = self.processing_duration_ticks # Fallback, or handle as error
        else:
            migration_info = task_to_provision._Service__migrations[-1]
            provisioning_latency_ticks = (
                migration_info["waiting_time"] +
                migration_info["pulling_layers_time"] +
                migration_info["migrating_service_state_time"]
            )
        
        provisioning_latency_seconds = provisioning_latency_ticks * self.simulator.tick_duration * 0.001 # tick_duration is in ms
        latency += provisioning_latency_seconds

        if chosen_server == self.fog_node:
            link = NetworkLink.find_by_id(1) # Link between Edge's BS and Fog's BS
            if link:
                network_hop_delay_seconds = link.delay * 0.001 # link.delay is in ms
                latency += network_hop_delay_seconds
            else:
                # Handle case where link is not found, though it should be based on setup
                print("Warning: NetworkLink ID 1 not found for latency calculation.")


        # 4. Calculate Energy Consumption
        total_energy_consumed_joules = 0.0
        duration_seconds = self.processing_duration_ticks * self.simulator.tick_duration * 0.001 # tick_duration is in ms
        
        # Energy for the chosen server during the processing window
        # get_power_consumption() returns current power in Watts.
        # We need average power over the duration or energy consumed during these ticks.
        # EdgeSimPy's power model usually tracks total energy consumed (server.power_model.energy_consumed).
        # A more accurate way would be to get energy before and after the self.processing_duration_ticks simulation steps.
        # However, the current structure calls simulator.step() which itself calls the power model's update.
        # For simplicity, let's approximate using the power state at the end of the processing.
        # This is a simplification. A better approach would be to sum `server.get_power_consumption() * self.simulator.tick_duration * 0.001`
        # for each tick within the loop, IF the power model updates per tick and not just on events.
        # Or, even better, if EdgeSimPy's power model has a way to query energy for the last X ticks.
        # Given LinearPowerModel, power is related to load. Load changes when task starts.
        # Let's assume get_power_consumption() gives the current power considering the new task.
        
        power_watts = chosen_server.get_power_consumption()
        total_energy_consumed_joules += power_watts * duration_seconds

        # 5. Define Reward
        # Negative rewards because we want to minimize latency and energy
        reward = self.w_latency * (-latency) + self.w_energy * (-total_energy_consumed_joules)

        # 6. Generate Next Task and Observation
        self.current_task = self._generate_task()
        next_observation = self._get_observation()

        # 7. Termination (terminated)
        terminated = False # Continuous environment

        # 8. Truncation (truncated)
        self.current_step_in_episode += 1
        truncated = self.current_step_in_episode >= self.max_steps_per_episode
        
        # If truncated, it's good practice to also clean up the last processed task's resources from the server
        # as it won't be "naturally" replaced by a new decision in the next step of this episode.
        # However, the reset method already handles cleaning up all services.
        # For a more fine-grained cleanup specific to this step if truncated:
        if truncated:
             if task_to_provision.server: # Check if it was successfully provisioned
                task_to_provision.server.deallocate_service(task_to_provision)


        return next_observation, reward, terminated, truncated, {}


    def render(self, mode='human'):
        # Placeholder for rendering logic
        pass

    def close(self):
        # Placeholder for cleanup logic
        pass

if __name__ == '__main__':
    # Test script (optional, can be added later)
    env = TaskOffloadingEnv()
    print("Environment initialized.")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Edge Device: {env.edge_device}")
    print(f"Fog Node: {env.fog_node}")
    if env.edge_device:
        print(f"Edge Device Power Model: {env.edge_device.power_model}")
    if env.fog_node:
        print(f"Fog Node Power Model: {env.fog_node.power_model}")

    # Further checks
    print(f"Simulator tick: {env.simulator.tick_duration} {env.simulator.tick_unit}")
    if env.edge_device and hasattr(env.edge_device, 'power_model'):
        assert env.edge_device.power_model.idle_power_consumption == 10
    if env.fog_node and hasattr(env.fog_node, 'power_model'):
        assert env.fog_node.power_model.idle_power_consumption == 50
    print("Manual power model assignment seems correct.")

    # Check if components were loaded
    print(f"Number of Edge Servers: {len(EdgeServer.all())}")
    print(f"Number of Base Stations: {len(BaseStation.all())}")
    print(f"Number of Network Links: {len(NetworkLink.all())}")
    print(f"Number of Network Switches: {len(NetworkSwitch.all())}")
    print(f"Number of Container Images: {len(ContainerImage.all())}")
    print(f"Number of Applications: {len(Application.all())}")


    assert len(EdgeServer.all()) == 2
    assert len(BaseStation.all()) == 2
    assert len(NetworkLink.all()) == 1
    assert len(NetworkSwitch.all()) == 1
    assert len(ContainerImage.all()) == 1
    assert len(Application.all()) == 1
    print("Dataset loading seems correct.")

    print("\nInitialization Test Passed!")

    # Test reset method
    print("\nTesting reset method...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Info: {info}")
    assert env.current_step_in_episode == 0
    assert "edge_cpu_load" in obs
    assert "fog_cpu_load" in obs
    assert "task_cpu_demand" in obs
    assert "network_latency_to_fog" in obs
    assert obs["edge_cpu_load"][0] == 0  # Initially no load
    assert obs["fog_cpu_load"][0] == 0   # Initially no load
    assert (50 / env.max_task_cpu_demand) <= obs["task_cpu_demand"][0] <= 1.0
    assert obs["network_latency_to_fog"][0] == (5 / env.max_network_latency) # Link delay is 5ms

    # Check if a task was generated
    assert env.current_task is not None
    print(f"Generated task: {env.current_task.label}, CPU: {env.current_task.cpu_demand}")
    assert env.current_task.application == env.application
    
    # Check simulator state after reset
    assert env.simulator.schedule.steps == 0
    assert env.simulator.schedule.time == 0
    assert len(Service.all()) == 1 # Only the current_task should exist as a service instance
    
    print("Reset method test passed!")

    # Test step method
    print("\nTesting step method (offload to edge)...")
    action_edge = np.array([-1.0], dtype=np.float32) # Offload to edge
    next_obs, reward, terminated, truncated, info = env.step(action_edge)
    
    print(f"Next Observation: {next_obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    assert env.current_step_in_episode == 1
    assert Service.all()[0].server == env.edge_device # First task (from reset) should be on edge
    assert len(Service.all()) == 2 # Old task + new current_task
    
    # Check if the new current_task is different from the one that was processed
    assert env.current_task.label != Service.all()[0].label


    print("\nTesting step method (offload to fog)...")
    action_fog = np.array([1.0], dtype=np.float32) # Offload to fog
    # Capture the task that will be processed
    task_to_be_processed_on_fog = env.current_task
    next_obs, reward, truncated, terminated, info = env.step(action_fog) # Swapped terminated and truncated for correct return order
    
    print(f"Next Observation: {next_obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}") # Should be False
    print(f"Truncated: {truncated}") # Should be False (unless max_steps is very small)
    print(f"Info: {info}")

    assert env.current_step_in_episode == 2
    # The task_to_be_processed_on_fog should now be on the fog_node.
    # It will be at Service.all()[1] if the previous edge task is still [0]
    # or find it by its label if order is not guaranteed.
    processed_task_on_fog = next((s for s in Service.all() if s.label == task_to_be_processed_on_fog.label), None)
    assert processed_task_on_fog is not None
    assert processed_task_on_fog.server == env.fog_node
    assert len(Service.all()) == 3 # Two old tasks + new current_task

    print("Step method basic tests passed!")
    
    # Test truncation
    print("\nTesting truncation...")
    env.max_steps_per_episode = 2 # Set for quick test
    env.reset()
    assert env.current_step_in_episode == 0
    env.step(action_edge) # Step 1
    assert env.current_step_in_episode == 1
    assert not truncated # First step should not truncate
    next_obs, reward, terminated, truncated, info = env.step(action_edge) # Step 2
    assert env.current_step_in_episode == 2
    assert truncated # Second step should truncate
    print("Truncation test passed!")
