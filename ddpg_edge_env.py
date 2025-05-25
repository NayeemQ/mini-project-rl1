import gymnasium
import numpy as np
import json # Added for saving temp config file
import os # Added for temp file path
from edge_sim_py import Simulator
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.network_link import NetworkLink
from edge_sim_py.components.base_station import BaseStation
from edge_sim_py.components.network_switch import NetworkSwitch
from edge_sim_py.components.power_models.servers import LinearServerPowerModel 
from edge_sim_py.components.service import Service 
from edge_sim_py.components.application import Application 
from edge_sim_py.components.container_image import ContainerImage 
from edge_sim_py.components.user import User 
from edge_sim_py.components import ContainerLayer, ContainerRegistry
from edge_sim_py.components.network_flow import NetworkFlow 
from edge_sim_py.components.topology import Topology 

class TaskOffloadingEnv(gymnasium.Env):
    def __init__(self, config=None):
        super().__init__()

        # 2.a. Initialize Simulator
        self.simulator = Simulator(
            tick_duration=1, 
            tick_unit="milliseconds",
            resource_management_algorithm=None  # DDPG agent will handle decisions
        )

        # 2.b. Define the dataset dictionary
        dataset = {
            "EdgeServer": [
                {
                    "attributes": {"id": 1, "model_name": "EdgeDevice", "cpu": 1000, "memory": 4096, "disk": 102400},
                    "relationships": {"power_model": "LinearServerPowerModel", "base_station": {"class": "BaseStation", "id": 1}}
                },
                {
                    "attributes": {"id": 2, "model_name": "FogNode", "cpu": 4000, "memory": 16384, "disk": 512000},
                    "relationships": {"power_model": "LinearServerPowerModel", "base_station": {"class": "BaseStation", "id": 2}}
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
            ],
            # Empty lists for other components to ensure proper initialization
            "User": [],
            "ContainerLayer": [],
            "ContainerRegistry": [],
            "NetworkFlow": [],
            "Topology": [],
            "Service": [] # Services will be created dynamically
        }

        # 2.c. Define the path for a temporary configuration file
        temp_config_path = "temp_sim_config.json"

        # 2.d. Write the dataset dictionary to this JSON file
        with open(temp_config_path, 'w') as f:
            json.dump(dataset, f, indent=4)

        # 2.e. Initialize the simulator using the file
        self.simulator.initialize(input_file=temp_config_path)

        # 2.f. Optionally, remove the temporary file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        # 2.g. Manually assign power models
        for server in EdgeServer.all():
            if server.id == 1:  # EdgeDevice
                server.power_model = LinearServerPowerModel(power_per_unit_load=0.5, idle_power_consumption=10)
            elif server.id == 2:  # FogNode
                server.power_model = LinearServerPowerModel(power_per_unit_load=0.7, idle_power_consumption=50)

        # 2.h. Define action space
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 2.i. Define observation space
        self.observation_space = gymnasium.spaces.Dict({
            "edge_cpu_load": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "fog_cpu_load": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "task_cpu_demand": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "network_latency_to_fog": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        # 2.j. Store references
        self.edge_device = EdgeServer.find_by_id(1)
        self.fog_node = EdgeServer.find_by_id(2)

        # 2.k. Initialize attributes
        self.task_counter = 0
        self.current_task = None  # This will hold the current Service object
        self.processing_duration_ticks = 10
        self.w_latency = 1.0
        self.w_energy = 0.5
        self.max_steps_per_episode = 100
        self.max_task_cpu_demand = 500.0  # MIPS, for normalization
        self.max_network_latency = 10.0  # ms, for normalization

        # 2.l. Store application reference
        self.application = Application.find_by_id(1)
        
        # Placeholder for current step in episode, typically reset in reset()
        self.current_step_in_episode = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles seeding for reproducibility

        # 1. Clean up existing services from the simulation
        # Need to iterate over a copy of the list if modifying it
        services_to_remove = list(Service.all()) 
        for service_instance in services_to_remove:
            if service_instance.server:
                # Decrement server's resource demand
                service_instance.server.cpu_demand -= service_instance.cpu_demand
                service_instance.server.memory_demand -= service_instance.memory_demand
                # Remove service from server's list
                if service_instance in service_instance.server.services:
                    service_instance.server.services.remove(service_instance)
            
            # Remove service from the simulator's schedule
            if service_instance in self.simulator.schedule.agents:
                self.simulator.schedule.remove(service_instance)
            
            # Remove service from global list of instances (if EdgeSimPy requires manual cleanup)
            if service_instance in Service._instances:
                Service._instances.remove(service_instance)

        # 2. Reset server demands (ensure they are clean)
        for server in EdgeServer.all():
            server.cpu_demand = 0
            server.memory_demand = 0
            server.disk_demand = 0 # If disk usage by tasks is modeled
            server.ongoing_migrations = 0 # Reset migrations if any

        # 3. Reset simulator's clock and step counter
        self.simulator.schedule.steps = 0
        self.simulator.schedule.time = 0
        self.simulator.last_dump = 0 # If metrics dumping is used

        # 4. Reset environment-specific counters
        self.task_counter = 0
        self.current_step_in_episode = 0

        # 5. Generate the first task for the new episode
        self.current_task = self._generate_task()

        # 6. Get the initial observation
        observation = self._get_observation()

        # 7. Return observation and an empty info dictionary
        return observation, {}

    def _generate_task(self):
        # Dummy task generation, will be properly implemented later
        self.task_counter += 1
        task_cpu = np.random.randint(50, int(self.max_task_cpu_demand) + 1) if self.max_task_cpu_demand > 50 else 50.0
        
        service = Service(
            label=f"task_{self.task_counter}",
            image_digest="dummy_image_digest",
            cpu_demand=task_cpu,
            memory_demand=128 
        )
        if self.application: # Check if self.application is not None
             service.application = self.application
        return service

    def _get_observation(self):
        # Dummy observation calculation, will be properly implemented later
        edge_load = np.float32(self.edge_device.cpu_demand / self.edge_device.cpu) if self.edge_device and self.edge_device.cpu > 0 else np.float32(0)
        fog_load = np.float32(self.fog_node.cpu_demand / self.fog_node.cpu) if self.fog_node and self.fog_node.cpu > 0 else np.float32(0)
        task_demand = np.float32(self.current_task.cpu_demand / self.max_task_cpu_demand) if self.current_task and self.max_task_cpu_demand > 0 else np.float32(0)
        
        link = NetworkLink.find_by_id(1)
        net_lat = np.float32(link.delay / self.max_network_latency) if link and self.max_network_latency > 0 else np.float32(0)

        return {
            "edge_cpu_load": np.clip(edge_load, 0, 1).reshape(1),
            "fog_cpu_load": np.clip(fog_load, 0, 1).reshape(1),
            "task_cpu_demand": np.clip(task_demand, 0, 1).reshape(1),
            "network_latency_to_fog": np.clip(net_lat, 0, 1).reshape(1)
        }

    def step(self, action):
        self.current_step_in_episode += 1

        # 1. Interpret Action
        # Assuming action is a single float from DDPG, e.g., in [-1, 1]
        # Threshold can be 0: < 0 for local (edge), >= 0 for fog
        decision_threshold = 0.0
        if action[0] < decision_threshold:
            chosen_server = self.edge_device
        else:
            chosen_server = self.fog_node

        task_to_provision = self.current_task
        
        # Ensure task has an application context if not already set (should be by _generate_task)
        if not task_to_provision.application and self.application:
            task_to_provision.application = self.application

        # 2. Simulate Task Provisioning and Execution
        # Ensure the service is registered with the simulator if it's brand new and not yet known by the scheduler
        # However, service.provision() should handle necessary registrations or the service should be an agent already.
        # If service is created fresh and not an agent, it might need self.simulator.initialize_agent(task_to_provision)
        # before provisioning. Let's assume .provision() handles this or it's not needed if service is already an agent.
        
        task_to_provision.provision(target_server=chosen_server)

        # Run the simulator for a fixed number of steps to simulate task processing
        # and allow provisioning to complete.
        # processing_duration_ticks is defined in __init__
        for _ in range(self.processing_duration_ticks):
            self.simulator.step() # Advances simulation by self.simulator.tick_duration

        # 3. Calculate Latency
        latency_seconds = 0.0
        if task_to_provision._Service__migrations: # Check if there are any migration records
            migration_info = task_to_provision._Service__migrations[-1]
            provisioning_latency_ticks = migration_info["waiting_time"] + \
                                         migration_info["pulling_layers_time"] + \
                                         migration_info["migrating_service_state_time"]
            # Convert simulator ticks (which are in milliseconds by our __init__) to seconds
            latency_seconds += provisioning_latency_ticks * self.simulator.tick_duration * 0.001 
        
        if chosen_server == self.fog_node:
            link = NetworkLink.find_by_id(1) # Assumes link ID 1 connects edge and fog base stations
            if link:
                latency_seconds += link.delay * 0.001 # link.delay is in ms, convert to seconds

        # 4. Calculate Energy Consumption
        # Energy for the chosen server during the processing ticks
        energy_consumed_joules = 0.0
        # duration_seconds is processing_duration_ticks * tick_duration (in ms) * 0.001
        duration_seconds = self.processing_duration_ticks * self.simulator.tick_duration * 0.001
        if chosen_server:
            power_watts = chosen_server.get_power_consumption()
            energy_consumed_joules += power_watts * duration_seconds
        
        # 5. Define Reward (weights w_latency, w_energy are in __init__)
        reward = self.w_latency * (-latency_seconds) + self.w_energy * (-energy_consumed_joules)

        # 6. Deallocate/Cleanup the completed task (important for resource management)
        # This mimics the task finishing and freeing up resources.
        if chosen_server and task_to_provision in chosen_server.services:
            # Use the server's deallocate_service method if available and it handles all cleanup
            # chosen_server.deallocate_service(task_to_provision) 
            # Manual cleanup if deallocate_service is not sufficient or for clarity:
            chosen_server.cpu_demand -= task_to_provision.cpu_demand
            chosen_server.memory_demand -= task_to_provision.memory_demand
            chosen_server.services.remove(task_to_provision)

        if task_to_provision in self.simulator.schedule.agents:
            self.simulator.schedule.remove(task_to_provision)
        
        # Check if the instance exists in the global list before removing
        # This can be problematic if Service.all() is not what's expected or if instance tracking is complex
        # For safety, only remove if clearly still present and not handled by other mechanisms.
        # A more robust way might be to have EdgeSimPy handle all deregistration via a single method call.
        if task_to_provision in Service.all(): # Service.all() returns a list copy
            Service._instances.remove(task_to_provision) # Accessing protected member _instances is not ideal

        # 7. Generate Next Task and Observation
        self.current_task = self._generate_task()
        next_observation = self._get_observation()

        # 8. Termination and Truncation
        terminated = False # Environment runs continuously unless a specific condition is met
        truncated = self.current_step_in_episode >= self.max_steps_per_episode
        
        if truncated:
            # If truncated, also clean up the newly generated current_task as it won't be processed.
            # This step might be optional or handled by reset() in the next episode.
            # For now, let reset handle the cleanup of any task present at the start of an episode.
            pass

        return next_observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Minimal test logic to ensure __init__ can be called
if __name__ == '__main__':
    print("Attempting to initialize TaskOffloadingEnv...")
    try:
        env = TaskOffloadingEnv()
        print("TaskOffloadingEnv initialized successfully.")
        print(f"Simulator object: {env.simulator}")
        print(f"Edge Device: {env.edge_device}, Power Model: {env.edge_device.power_model if env.edge_device else 'N/A'}")
        print(f"Fog Node: {env.fog_node}, Power Model: {env.fog_node.power_model if env.fog_node else 'N/A'}")
        print(f"Action Space: {env.action_space}")
        print(f"Observation Space: {env.observation_space}")

        # Test if dataset components were loaded
        print(f"Number of Edge Servers: {len(EdgeServer.all())}")
        assert len(EdgeServer.all()) == 2, "Should have 2 edge servers"
        print(f"Number of Applications: {len(Application.all())}")
        assert len(Application.all()) == 1, "Should have 1 application"
        
        # Verify power model assignment
        if env.edge_device and hasattr(env.edge_device, 'power_model'):
             assert isinstance(env.edge_device.power_model, LinearServerPowerModel), "Edge device power model type mismatch"
             assert env.edge_device.power_model.idle_power_consumption == 10, "Edge device power model params mismatch"
        if env.fog_node and hasattr(env.fog_node, 'power_model'):
            assert isinstance(env.fog_node.power_model, LinearServerPowerModel), "Fog node power model type mismatch"
            assert env.fog_node.power_model.idle_power_consumption == 50, "Fog node power model params mismatch"
        print("Power model assignment seems correct.")

        print("__init__ method test completed.")

    except Exception as e:
        print(f"Error during TaskOffloadingEnv initialization or test: {e}")
        import traceback
        traceback.print_exc()
