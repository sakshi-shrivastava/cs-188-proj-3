# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        #print("position: " + str(newPos))
        #print("score: " + str(successorGameState.getScore()))
        #print(type(newGhostStates))

        gridSize = newFood.width * newFood.height

        ghostDists = []
        for ghostState in newGhostStates:
            #print(ghostState)
            """for coord in successorGameState.getGhostPositions(ghostState.index):
                dist = manhattanDistance(newPos, coord)
                ghostDists.append(dist)"""
            ghostCoord = ghostState.configuration.pos #coordinates of ghost
            dist = manhattanDistance(newPos, ghostCoord)
            ghostDists.append(dist)
        ghostEval = min(ghostDists) #nearest ghost
        print(ghostDists)   
        print(ghostEval)
        ghostEval = -100 * gridSize * ghostEval #punish for having a ghost too close"""

        #newFoodEval = manhattanDistance(newFood.configuration.pos, newPos)
        #print(newFood)

        foodDists = []
        foodCount = 0
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:
                    foodDists.append(manhattanDistance(newPos, (x, y)))
                foodCount = foodCount + 1
        foodCountEval = -1 * 50 * foodCount #we want the number of foods left to make the evaluation lower
        foodDistEval = -1 * min(foodDists) #aim to get closer to food

        gettingPointsEval = successorGameState.getScore() #we want more points

        return ghostEval + foodCountEval + foodDistEval + gettingPointsEval
            

        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # def minimaxValues(state, depth, agent):
        #     if agent == state.getNumAgents():
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         else:
        #             return minimaxValues(state, depth + 1, 0)
        #     else:
        #         actions = state.getLegalActions(agent)

        #         if len(actions) == 0:
        #             return self.evaluationFunction(state)

        #         next_states = (
        #             minimaxValues(state.generateSuccessor(agent, action),
        #             depth, agent + 1)
        #             for action in actions
        #             )

        #         return (max if agent == 0 else min)(next_states)

        # return max(
        #     gameState.getLegalActions(0),
        #     key = lambda x: minimaxValues(gameState.generateSuccessor(0, x), 1, 1)
        #     )


        bestAction = None
        bestValue = -float('inf')
        for action in gameState.getLegalActions(0):
            nextValue = self.minimaxValues(gameState.getNextState(0, action), self.depth, 1)
            if nextValue > bestValue:
                bestAction = action
                bestValue = nextValue

        return bestAction

    def minimaxValues(self, gameState, depth, agentNum):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        if agentNum == 0:
            val = -float('inf')
            nextStates = [gameState.getNextState(0, action) for action in gameState.getLegalActions(0)]
            for nextState in nextStates:
                v2 = self.minimaxValues(nextState, depth, 1)
                val = max(val, v2)
        else:
            val = float('inf')
            nextStates = [gameState.getNextState(agentNum, action) for action in gameState.getLegalActions(agentNum)]
            for nextState in nextStates:
                if agentNum == gameState.getNumAgents() - 1:
                    v2 = self.minimaxValues(nextState, depth - 1, 0)
                else:
                    v2 = self.minimaxValues(nextState, depth, agentNum + 1)
                val = min(val, v2)
        return val




    #     bestScore,bestNextMove=self.minimax(self.index,gameState)
    #     return bestNextMove

    # def minimax(self,agent,gameState,depth=0):
    #     # Pacman agent is the maximizer (zero value)
    #     # Ghosts agents are minimizers (values greater than zero)
    #     bestAction=None
    #     if (depth==self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose()):
    #         return [self.evaluationFunction(gameState),bestAction]
    #     if (agent==0): # pacman agent
    #         return self.maxvalue(agent,gameState,depth)
    #     else:          # ghosts agents
    #         return self.minvalue(agent,gameState,depth)

    # def maxvalue(self,maximizingPlayer,gameState,depth):
    #     bestAction=None
    #     best=-float('inf')
    #     for succ in gameState.getLegalActions(maximizingPlayer):
    #         temp=best
    #         value=self.minimax((depth+1) % gameState.getNumAgents(),gameState.generateSuccessor(maximizingPlayer,succ),depth+1)[0]
    #         best=max(best,value)
    #         if temp!=best:
    #             bestAction=succ
    #     return [best,bestAction]

    # def minvalue(self,minimizingPlayer,gameState,depth):
    #     bestAction=None
    #     best=float('inf')
    #     for succ in gameState.getLegalActions(minimizingPlayer):
    #         temp=best
    #         value=self.minimax((depth+1) % gameState.getNumAgents(),gameState.generateSuccessor(minimizingPlayer,succ),depth+1)[0]
    #         best=min(best,value)
    #         if temp!=best:
    #             bestAction=succ
    #     return [best,bestAction]
    



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
