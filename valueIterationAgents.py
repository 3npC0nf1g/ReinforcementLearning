# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import sys

import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        self.best_action_dict = {}  # On crée un dictionnaire pour stocker la meilleure action
        for i in range(iterations):  # Pour une valeur dans l'ensemble d'itération on exécute la boucle
            nextvalues = self.values.copy()  # Création d'une copie de l'état actuel des valeurs
            for s in mdp.getStates(): # On parcours tous les états possibles du MDP
                actions = mdp.getPossibleActions(s) # On récupère toutes les actions possibles
                if len(actions) > 0: # On vérifie qu'au moins une action est disponible
                  bestaction = None # On initiale la meilleure action
                  bestscore = -sys.float_info.max # On initialise le meilleur score, on part sur du moins l'infini
                  for a in actions:  # On parcours chaque action
                    E = self.getQValue(s,a) # On calcul pour chaque action la Q valeur
                    if E > bestscore:      # On vérifie si la Q valeur calculer est meilleure que le score actuel
                      bestscore = E       # On met à jour le meilleur score
                      bestaction = a       # On met à jour la meilleure action correspondante
                  self.best_action_dict[s] = bestaction # On stocke l'action associée à la meilleure valeur pour l'état s
                  nextvalues[s] = bestscore # Mise à jour de la meilleure valeur trouvée dans la copie de valeur pour l'état actuel
            self.values = nextvalues # Après avoir calculer la meilleure valeur trouvée pour tous les états on met à jour les valeurs actuelles avec celles de la copie


    def getValue(self, state):
        """
      Return the value of the state (computed in __init__).
    """
        return self.values[state] # Retourne la meilleure valeure corresponde à l'état actuel, obtenue après l'itération des valeurs

    def getQValue(self, state, action):
        """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
        stateprobPaires = self.mdp.getTransitionStatesAndProbs(state, action)
        E = 0.0
        for ss, T in stateprobPaires:
          if ss in self.values: #Vérifier si l'état suivant a une valeur
            E += T *(self.mdp.getReward(state,action,ss) + self.discount*self.values[ss])
          else:
            E += T * self.mdp.getReward(state, action, ss) #Si c'est un état terminal, la valeur est nulle
        return E


    def getPolicy(self, state): #  Retourne l'action optimale dans un état donné telle que déterminée pendant l'itération de valeurs
        """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
        return self.best_action_dict.get(state)


    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
