from collections import deque
from constants import *
from environment import *
from state import State
import numpy as np

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

Last updated by njc 08/09/22
"""


class node:
    def __init__(self, state, actions):
        self._state = state
        self._actions = actions


class Solver:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.probabilities = []
        self.rewards = []
        self.policy = {}
        self.policyArr = []
        self.exploredStates = {}
        self.arrStates = []
        self.intialState = self.environment.get_init_state()
        self.maxDiff = 0
        self.converged = False
        # TODO: Define any class instance variables you require (e.g. dictionary mapping state to VI value) here.
        #
        pass

    # === Value Iteration ==============================================================================================
    def getActionResAndProb(self, action):
        # print("fetching probabilities")
        double = self.environment.double_move_probs[action]
        ccw = self.environment.drift_ccw_probs[action]
        cw = self.environment.drift_cw_probs[action]
        pdouble = (1 - cw - ccw) * double
        pcw = cw * (1 - double)
        pccw = ccw * (1 - double)
        pdccw = ccw * double
        pdcw = cw * double
        success = (1 - ccw - cw) * (1 - double)
        movementsAndProbs = {
            "CW": pcw,
            "CCW": pccw,
            "DCW": pdcw,
            "DCCW": pdccw,
            "D": pdouble,
            "S": success,
        }
        movements = {
            "CW": [SPIN_RIGHT, action],
            "CCW": [SPIN_LEFT, action],
            "DCW": [SPIN_RIGHT, action, action],
            "DCCW": [SPIN_LEFT, action, action],
            "D": [action, action],
            "S": [action],
        }
        return movementsAndProbs, movements

    def calcRewards(self, state, movements):
        # print("Calculating Rewards")
        minReward = 0
        newState = state
        for movement in movements:
            reward, newState = self.environment.apply_dynamics(
                newState, movement)
            if reward < minReward:
                min_reward = reward
        return min_reward, newState

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self.exploredStates = {}
        self.policy = {}
        self.arrStates = [self.intialState]
        self.exploredStates[self.intialState] = 0
        nodeStack = deque()
        nodeStack.append(self.intialState)
        while nodeStack:
            currentState = nodeStack.pop()
            self.exploredStates[currentState] = 0
            self.policy[currentState] = FORWARD
            for action in ROBOT_ACTIONS:
                reward, newState = self.environment.apply_dynamics(
                    currentState, action)
                if newState not in self.exploredStates:
                    self.arrStates.append(newState)
                    nodeStack.append(newState)

    def vi_is_converged(self):
        if self.maxDiff < self.environment.epsilon and self.maxDiff != 0:
            return True
        return False

    def vi_iteration(self):
        maxdiff = 0
        tempNewStates = dict(self.exploredStates)

        for state in self.exploredStates:
            if self.environment.is_solved(state):
                self.exploredStates[state] = 0.0
                continue
            tempActionVals = dict()
            for action in ROBOT_ACTIONS:
                actionVal = 0
                movementAndProbs, movements = self.getActionResAndProb(action)
                for movementType in movements:
                    reward, newState = self.calcRewards(
                        state, movements[movementType])
                    if self.environment.is_solved(newState):
                        reward = 0
                    actionVal += movementAndProbs[movementType] * (
                        reward + self.environment.gamma *
                        self.exploredStates[newState]
                    )
                tempActionVals[action] = actionVal

            self.exploredStates[state] = max(tempActionVals.values())
            self.policy[state] = max(tempActionVals, key=tempActionVals.get)
            if abs(tempNewStates[state] - self.exploredStates[state]) > maxdiff:
                maxdiff = abs(tempNewStates[state] -
                              self.exploredStates[state])

        self.maxDiff = maxdiff

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while not self.vi_is_converged():
            self.vi_iteration()

    def vi_get_state_value(self, state: State):
        return self.exploredStates[state]

    def vi_select_action(self, state: State):
        return self.policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        self.vi_initialise()
        self.arrStates = list(self.exploredStates.keys()) # states
        numOfStates = len(self.exploredStates) 
        numOfActions = len(ROBOT_ACTIONS)
        self.probabilities = np.zeros([numOfStates, numOfActions, numOfStates]) # t_model
        self.rewards = np.zeros([numOfStates, numOfActions]) # r_model
        self.find_state_parameters(self.arrStates, ROBOT_ACTIONS)
        self.policyArr = np.zeros([numOfStates], dtype=np.int64) + FORWARD

    def pi_is_converged(self):
        return self.converged

    def pi_iteration(self):
        self.policy_evaluation()
        self.policy_improvement()

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()

    def pi_select_action(self, state: State):
        return self.policy[state]

    def find_state_parameters(self, states, actions):

        for state in states:
            stateInd = states.index(state)
            if self.environment.is_solved(state):
                self.probabilities[stateInd][:][:] = 0
                self.rewards[stateInd][:] = 0
                continue
            for action in actions:
                rewardSum = 0
                movementAndProbs, movements = self.getActionResAndProb(action)
                for movementType in movements:
                    reward, newState = self.calcRewards(
                        state, movements[movementType])
                    self.probabilities[stateInd][action][
                        self.arrStates.index(newState)
                    ] += movementAndProbs[movementType]
                    rewardSum += reward * movementAndProbs[movementType]

                self.rewards[stateInd][action] = rewardSum

    def policy_evaluation(self):
        numOfStates = len(self.arrStates)
        self.policyArr = list(self.policy.values()) 
        stateIndicies = np.array(range(numOfStates))  # state numbers

        policyStateProbablilities = self.probabilities[stateIndicies,
                                                       self.policyArr] # t-pi
        policyStateRewards = self.rewards[stateIndicies, self.policyArr]
        Vs = np.linalg.solve(
            np.identity(numOfStates)
            - (self.environment.gamma * policyStateProbablilities),
            policyStateRewards,
        )
        self.exploredStates = {
            state: Vs[index] for index, state in enumerate(self.arrStates)
        }

    def policy_improvement(self):
        tempOldPolicy = dict(self.policy)
        for state in self.exploredStates:
            if self.environment.is_solved(state):
                self.exploredStates[state] = 0.0
                continue
            tempActionVals = dict()
            for action in ROBOT_ACTIONS:
                actionVal = 0 # total
                movementAndProbs, movements = self.getActionResAndProb(action)
                for movementType in movements:
                    reward, newState = self.calcRewards(
                        state, movements[movementType])
                    if self.environment.is_solved(newState):
                        reward = 0
                    actionVal += movementAndProbs[movementType] * (
                        reward + self.environment.gamma *
                        self.exploredStates[newState]
                    )
                tempActionVals[action] = actionVal

            self.policy[state] = max(tempActionVals, key=tempActionVals.get)

        if all(self.policy[s] == tempOldPolicy[s] for s in self.arrStates):
            self.converged = True
