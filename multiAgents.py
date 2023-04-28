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
        distance = float("inf")
        if successorGameState.isWin():
            return distance

      
        newFoodNext = newFood.asList()
        currentNumFood = currentGameState.getNumFood()
        successorNumFood = successorGameState.getNumFood()
        successorScore = successorGameState.getScore()
        distance = float("-Inf")
        #succsessor = successorScore - 2
        #newTime = min(newScaredTimes)

        for state in newGhostStates:
            getDistance = state.getPosition()
            GhostmanhattanDis = manhattanDistance(getDistance, newPos)
            if GhostmanhattanDis is 0.5:
                return distance

        
        food_Distance = min(manhattanDistance(newPos, food) for food in (newFoodNext))

        successor_f = 0
        if (currentNumFood > successorNumFood):
            successor_f = 0
            successor_f = 90

        reflex = successorScore - 2 * food_Distance + successor_f

        return reflex
       
       
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

       
           
        def evalComparison(agentIndex, eval1, eval2):
            if eval1 == None: #none checks for when we pass in a default best action
                return eval1
            elif eval2 == None:
                return eval2
            if agentIndex == 0:
                if eval1[0] > eval2[0]:
                    return eval1
                else:
                    return eval2
            else:
                if (type(eval1[0]) != float) or (type(eval1[0]) != int):
                    return eval2
                if (type(eval2[0]) != float) or (type(eval2[0]) != int):
                    return eval1
                if eval1[0] < eval2[0]:
                    return eval1
                else:
                    return eval2

        
        def getActionHelper(state, agentIndex, depth, prevAction):

            if state.isWin() or state.isLose() or depth <= 0:
                #print("appending:")
                ret = (self.evaluationFunction(state), prevAction)# (self.evaluationFunction(state), prevAction)
                #print(ret)
                return ret
            
            if agentIndex == 0:
                scores = []
                actions = state.getLegalActions(agentIndex)
                
                for action in actions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    scores.append(getActionHelper(successorState, 1, depth, action))
                #print(actions)
                #print(scores)

                best = actions[0]
                bestScore = scores[0][0]

                for index in range(0, len(actions)):
                    #print("evaluating:")
                    #print(scores[index][0])
                    #print(actions[index])
                    #print("vs")
                    #print(bestScore)
                    #print(best)
                    #print("depth:")
                    #print(depth)
                    if scores[index][0] > bestScore:
                        bestScore = scores[index][0]
                        best = actions[index]

            else:
                scores = []
                actions = state.getLegalActions(agentIndex)

                nextAgentIndex = agentIndex + 1
                if nextAgentIndex >= state.getNumAgents():
                    nextAgentIndex = 0
                    depth = depth - 1
                
                for action in actions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    scores.append(getActionHelper(successorState, nextAgentIndex, depth, action))
                #print(actions)
                #print(scores)

                best = actions[0]
                bestScore = scores[0][0]

                for index in range(0, len(scores)):
                    #print("evaluating:")
                    #print(scores[index][0])
                    #print(actions[index])
                    #print("vs")
                    #print(bestScore)
                    #print(best)
                    #print("depth:")
                    #print(depth)
                    if scores[index][0] < bestScore:
                        bestScore = scores[index][0]
                        best = actions[index]

            return (bestScore, best)
            

        return (getActionHelper(gameState, 0, self.depth, Directions.STOP))[1]


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

        #def getActionHelper(state, depth)
       
        
       
        



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def getActionHelper2(state, agentIndex, depth, prevAction, alpha, beta):

            if agentIndex >= state.getNumAgents():
                agentIndex = 0
                depth = depth - 1

            if state.isWin() or state.isLose() or depth <= 0:
                #print("appending:")
                ret = (self.evaluationFunction(state), prevAction, alpha, beta)# (self.evaluationFunction(state), prevAction)
                #print(ret)
                return ret

            actions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:
                maxScore = float("-inf")
                maxAction = Directions.STOP

                #depth = depth - 1
                
                for action in actions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    successorScore, successorAction, a2, b2 = getActionHelper2(successorState, 1, depth, action, alpha, beta)
                    
                    if successorScore > maxScore:
                        maxScore = successorScore
                        maxAction = action

                    if maxScore > beta:
                        return (maxScore, maxAction, alpha, beta)

                    alpha = max(successorScore, alpha)

                return (maxScore, maxAction, alpha, beta)

            else:
                minScore = float("inf")
                minAction = actions[0]

                nextAgentIndex = agentIndex + 1
                if nextAgentIndex >= state.getNumAgents():
                    nextAgentIndex = 0
                    depth = depth - 1
                
                for action in actions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    successorScore, successorAction, a2, b2 = getActionHelper2(successorState, nextAgentIndex, depth, action, alpha, beta)

                    if successorScore < minScore:
                        minScore = successorScore
                        minAction = action

                    if minScore < alpha:
                        return (minScore, minAction, alpha, beta)

                    beta = min(beta, successorScore)
                #print(actions)
                #print(scores)

                #ret = (minScore, minAction, alpha, beta)
                #print(ret)

                return (minScore, minAction, alpha, beta)

        return (getActionHelper2(gameState, 0, self.depth, Directions.STOP, float("-inf"), float("inf")))[1]

        

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

        return self.getExpectedValue(gameState,0, self.depth)[1]

       
    def getExpectedValue(self, game_state,agent_index, depth):
        
        
        if game_state.isWin():
            return [self.evaluationFunction(game_state), None]
        if game_state.isLose():
            return [self.evaluationFunction(game_state), None]
        if depth == 0:
            return [self.evaluationFunction(game_state), None]
        if game_state.getLegalActions(agent_index) == 0:
           return [self.evaluationFunction(game_state), None]
           
       
        if agent_index is not 0:
            move_legal1 = game_state.getLegalActions(agent_index)
            val= agent_index
            ideal_action1 = Directions.STOP
            newIndex = agent_index + 1*1
            prob = 1.0 * float(len(move_legal1))
            #flo = float("inf")
            numAgents = game_state.getNumAgents()
            index1 = (agent_index + 1) % numAgents
            if newIndex == numAgents:
                depth = depth - 1
            for act in move_legal1:
                successorState = game_state.generateSuccessor(agent_index, act)
                value_expecti, _ = self.getExpectedValue(successorState, index1, depth)
                val = val + value_expecti/ prob
            return val, ideal_action1
            
           

        if agent_index is 0:
            Value_Max = float("-inf")
            value = Value_Max
            ideal_action2 = Directions.STOP
            move_legal2 = game_state.getLegalActions(0)
            index2 = (agent_index + 1) % game_state.getNumAgents()
            for act in move_legal2:
                successorState = game_state.generateSuccessor(0, act)
                pac_score, _ = self.getExpectedValue(successorState, index2, depth)
                if max(value, pac_score) == pac_score:
                   value, ideal_action2 = pac_score, act
            return (value, ideal_action2)
        
        #   util.raiseNotDefined()
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin(): 
        return float("inf")
    elif currentGameState.isLose():
        return float("-inf")

    pacmanPos = currentGameState.getPacmanPosition()
    
    ghostStates = currentGameState.getGhostStates()
    nearestGhostDist = float("inf")

    for ghostState in ghostStates:
        ghostCoords = ghostState.getPosition()
        currGhostDist = manhattanDistance(ghostCoords, pacmanPos)
        
        if ghostState.scaredTimer == 0:
            if nearestGhostDist > currGhostDist:
                nearestGhostDist = currGhostDist
        else:
            nearestGhostDist = -10

    foods = (currentGameState.getFood()).asList()
    nearestFoodDist = float("inf")
    #farthestFoodDist = float("-inf")
    numFoods = len(foods)

    if not foods: #there are no more foods
        nearestFoodDist = 0

    else:
        for food in foods:
            currFoodDist = manhattanDistance(food, pacmanPos)
            if nearestFoodDist > currFoodDist:
                nearestFoodDist = currFoodDist
            #if nearestFoodDist < currFoodDist:
            #    farthestFoodDist= currFoodDist

    scoreFactor = currentGameState.getScore() * 10
    ghostFactor = 5 / (nearestGhostDist + 1)
    foodFactor = (((nearestFoodDist) * -1) - numFoods) / 5

    return scoreFactor - ghostFactor + foodFactor
# Abbreviation
better = betterEvaluationFunction
