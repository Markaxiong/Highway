from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {    
                "observation": {
                    "type": "TimeToCollision",
                    #"vehicles_count": 7,
                   # "features": ["presence", "x", "y", "vx", "vy"],
                  #"normalize": False,
                 #   "absolute" : True
                },
                #"vehicles_count": 7,
                "action": {"type": "DiscreteMetaAction"},
                "simulation_frequency": 15,  # [Hz]
                "policy_frequency": 1,  # [Hz]
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "screen_width": 600,  # [px]
                "screen_height": 150,  # [px]
                "centering_position": [0.3, 0.5],
                "scaling": 5.5,
                "show_trajectories": False,
                "render_agent": True,
                "offscreen_rendering": False,
                "collision_reward": -1.5,
                "right_lane_reward": 0.2,
                "high_speed_reward": 0.2,
                "reward_speed_range": [2, 12],
                "merging_speed_reward": 0,
                "lane_change_reward": 1,
                "merging_time_penalty": -0.2,
                "success_reward": 2.0  # Reward for successful merge
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )

        return utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["merging_speed_reward"]+self.config["merging_time_penalty"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"]+self.config["success_reward"],
               
            ],
            [0, 1],
        )

    def _rewards(self, action: int) -> Dict[Text, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed,
            #"right_lane_reward": self.vehicle.lane_index[2] / 1,
            "right_lane_reward": self.vehicle.lane_index == ("b", "c", 1),
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 1)
                and isinstance(vehicle, ControlledVehicle)
            ),
            "merging_time_penalty" : sum(
            self.config["merging_time_penalty"]
            for vehicle in self.road.vehicles
            if vehicle.lane_index == ("b", "c", 0) and isinstance(vehicle, ControlledVehicle) 
        ),
            "success_reward" :self.vehicle.position[0] > 310
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        #print("crash" + str(self.vehicle.crashed))
        #print("over" + str(self.vehicle.position[0] > 370))
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        #ends = [150, 100, 100, 100]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                   [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
               "d",
               StraightLane(
                   [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
               ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road


    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            #road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=30
            road,road.network.get_lane(("j","k",0)).position(110 + self.np_random.uniform(-5,5),0), speed = 14 + self.np_random.uniform(-1,1)
        )
        ego_vehicle.target_speed = 15 + self.np_random.uniform(-1,1)

        if ego_vehicle.lane_index == ("b", "c", 0):
            ego_vehicle.enable_lane_change = True  # 允许变道到主路
        else:
            ego_vehicle.enable_lane_change = False  # 禁止其他变道
    
        road.vehicles.append(ego_vehicle)
         # 在状态更新时检查并应用变道约束
        def update_lane_change_permission():
            if ego_vehicle.lane_index == ("b", "c", 0):
                ego_vehicle.enable_lane_change = True  # 允许变道到主路
            else:
                ego_vehicle.enable_lane_change = False  # 禁止其他变道

        ego_vehicle.on_state_update = update_lane_change_permission
        
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        positions_speeds = [
            (140, 12.5), (150, 14), (160, 15.5), 
            (180, 14.5), (190, 17), (200, 20)
        ]

        for position, speed in positions_speeds:
            lane = road.network.get_lane(("a", "b", 1))
            position = lane.position(position + self.np_random.uniform(-2, 2), 0)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(
                other_vehicles_type(
                    road, position, heading=lane.heading_at(position), speed=speed, enable_lane_change=False
                )
            )

        #merging_v = other_vehicles_type(
            #road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20
        #)
        #merging_v.target_speed = 30
        #road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle
