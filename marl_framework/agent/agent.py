import logging
from typing import Dict


from actor.network import ActorNetwork
from agent.action_space import AgentActionSpace
from agent.communication_log import CommunicationLog
from agent.state_space import AgentStateSpace

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        actor_network: ActorNetwork,
        params: Dict,
        mapping,
        agent_id: int,
        agent_state_space: AgentStateSpace,
    ):
        self.params = params
        self.agent_id = agent_id
        self.mission_type = self.params["experiment"]["missions"]["type"]
        self.n_actions = self.params["experiment"]["constraints"]["num_actions"]
        self.v_max = self.params["experiment"]["uav"]["max_v"]
        self.a_max = self.params["experiment"]["uav"]["max_a"]
        self.x_dim = params["environment"]["x_dim"]
        self.y_dim = params["environment"]["y_dim"]
        self.mapping = mapping
        self.local_map = mapping.init_priors()
        self.agent_state_space = agent_state_space
        self.action_space = AgentActionSpace(self.params)
        self.actor_network = actor_network
        self.agent_info = dict()
        self.position = None
        self.map_footprint = None
        self.map2communicate = None

    def communicate(
        self, t, num_episode, communication_log: CommunicationLog, mode
    ) -> CommunicationLog:
        if t == 0:
            self.position = self.agent_state_space.get_random_agent_state(
                self.agent_id, num_episode
            )
            self.local_map, self.map_footprint, _, self.map2communicate, self.footprint_img = self.mapping.update_grid_map(
                self.position, self.local_map, t, mode
            )

        agent_info = {
            "local_map": self.local_map,
            "position": self.position,
            "map_footprint": self.map_footprint,
            "map2communicate": self.map2communicate,
            "footprint_img": self.footprint_img,
        }
        global_log = communication_log.store_agent_message(agent_info, self.agent_id)

        return global_log, self.local_map, self.position

    def receive_messages(self, communication_log, agent_id, t):
        # receive all available communication
        received_communication = communication_log.get_messages(self.agent_id)
        if len(received_communication) > 0:
            # fuse most certain map information into new map state
            self.local_map = self.mapping.fuse_map(
                self.local_map, received_communication, agent_id, "local"
            )

        return received_communication, self.local_map

    def step(self, agent_id, t, num_episode, batch_memory, mode, next_other_positions):
        # Get action mask for masking out currently invalid actions (outside of environment)

        action_mask_1d, _ = self.action_space.get_action_mask(self.position)
        action_mask_1d = self.action_space.apply_collision_mask(
            self.position, action_mask_1d, next_other_positions, self.agent_state_space
        )

        # action choice
        probs, action, mask, eps = self.actor_network.get_action_index(
            batch_memory, action_mask_1d, self.agent_id, t, num_episode, mode
        )
        # Append chosen action to previous position to get new position
        self.position = self.action_space.action_to_position(self.position, action)

        if not self.is_in_map(self.position):
            print("OUT OF MAP")

        # Sense and update grid map
        self.local_map, self.map_footprint, footprint_idx, self.map2communicate, self.footprint_img = self.mapping.update_grid_map(
            self.position, self.local_map, t, mode
        )
        batch_memory.insert(-1, agent_id, action=action, mask=mask)

        return (
            self.local_map,
            self.position,
            eps,
            action,
            footprint_idx,
            self.map2communicate,
        )

    def is_in_map(self, position):
        if (
            position[0] <= self.x_dim
            and position[0] >= 0
            and position[1] <= self.y_dim
            and position[1] >= 0
            and position[2] >= 5
            and position[2] <= 15
        ):
            return True
        else:
            return False
