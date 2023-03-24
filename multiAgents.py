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

        currPos = currentGameState.getPacmanPosition()
        gridSize = newFood.width * newFood.height

        foodCount = 0
        foodDists = []
        ghostDists = []
        minGhostDist = gridSize

        foodEval = 0
        ghostEval = 0


        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:
                    foodDists.append(manhattanDistance(currPos, (x, y)))
                foodCount = foodCount + 1
        foodCountEval = gridSize - foodCount #-1 * (foodCount ** 2) #we want the number of foods left to make the evaluation lower
        foodDistEval = 0
        if len(foodDists) > 0:
            foodDistEval =-1 * min(foodDists) #aim to get closer to food

        for ghostState in newGhostStates:
            if ghostState.scaredTimer <= 0:
                ghostCoord = ghostState.getPosition() #coordinates of ghost
                dist = manhattanDistance(newPos, ghostCoord)
                ghostDists.append(dist)
        if len(ghostDists) > 0:
            minGhost = min(ghostDists)
            if minGhost < 10:
                ghostEval = -1 * (minGhost ** 2)

        gettingPointsEval = successorGameState.getScore() #we
        #if gettingPointsEval < 0:
        #    gettingPointsEval= (gettingPointsEval ** 2) * -1

        #return ghostEval + foodCountEval + foodDistEval + gettingPointsEval + findFoodMotivator

        return ghostEval + foodEval + gettingPointsEval + foodCountEval + foodDistEval
            

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

        gameState.getLegalActions(agent Index):
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

       
           
          

        
        def maxValue(gameState,depth, agent):
          legalMov = gameState.getLegalActions(agent)
          
          if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
          if depth < 0 :
            return self.evaluationFunction(gameState), Directions.STOP  
          if depth == self.depth*gameState.getNumAgents() :
            return (None,self.evaluationFunction(gameState)) 
          

          v = float("-inf")

          for action in legalMov:
            newState = gameState.generateSuccessor(agent,action)
            score = minValue(newState,depth,agent+1)[0]
    
            if score > v:
              v = score
              maxAction = action
          return (v,maxAction)
        




        def minValue(gameState,depth,agent):
          legalAction = gameState.getLegalActions(agent)
          if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
          if depth < 0 :
            return self.evaluationFunction(gameState), Directions.STOP  
          if depth == self.depth*gameState.getNumAgents() :
            return (None,self.evaluationFunction(gameState)) 
          
          
          v = float("inf")
         
         # if agent < gameState.getNumAgents()-1
       

          for action in legalAction:
            successor = gameState.generateSuccessor(agent,action)
            score = maxValue(successor,depth,agent)[0]

            if score < v:
              v = score
              minAction = action          
          return (v,minAction)
        

        #return maxValue(gameState,self.depth,0)[1]
       
        



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
