# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qVals = util.Counter()  # Initialisation d'un attribut qVals pour stocker les Q-valeurs

    def getQValue(self, state, action):
        """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
        return self.qVals[(state, action)]  # Renvoit la Q-valeur associée à un tuple (état, action)

    def getValue(self, state):
        """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
        actions = self.getLegalActions(state)  # Renvoit d'une liste d'actions possibles dans un état donné
        if len(actions) == 0:  # Vérifie si la liste est vide
            return 0.0  # Si oui, renvoit directement 0.0
        values = []  # Création d'une liste vide pour stocker les Q-valeurs
        for action in actions:  # Parcourt de chaque action légale
            values.append(self.getQValue(state, action))  # Calcul de chaque Q-valeurs et ajout de celle-ci à values
        return max(values)  # Retour de la valeur maximale, correspondante à la meilleure estimation de la récompense
        # cumulée

    def getPolicy(self, state):
        """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
        actions = self.getLegalActions(state)  # Renvoit d'une liste d'actions possibles dans un état donné
        allActions = []  # Création d'une liste vide pour stocker des tuples constituer d'une Q- valeur et de l'action
        # liée à la valeur
        if len(actions) == 0:  # Vérifier si aucune action légale n'est disponible
            return None  # Si c'est le cas, retourner None
        for action in actions:  # Parcourt de chaque action légale
            allActions.append((self.getQValue(state, action), action))  # Pour chaque action donnée récupérer la
            # Q-valeur correspondant à l'état et l'action ensuite la rajouter à allActions
        bestActions = [pair for pair in allActions if
                       pair == max(allActions)]  # Trouver les actions qui ont la Q-valeur maximale
        bestActionPair = random.choice(
            bestActions)  # Si plusieurs actions ont la même Q-valeur, en choisir une de façon aléatoire
        return bestActionPair[1]  # Retourne l'action associée à la meilleur Q-valeur

    def getAction(self, state):
        """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
        # Pick Action
        legalActions = self.getLegalActions(state)  # Renvoit d'une liste d'actions possibles dans un état donné
        action = None  # Initialise action à None
        p = self.epsilon  # p est la probabilité associée à l'exploration
        if util.flipCoin(p):  # Simulation d'un lancer de piece
            action = random.choice(legalActions)  # Si le lancer renvoit True, l'agent explore, une action aléatoire
            # est choisit parmi les actions légales (ceci correspond à une exploration, c'est associé à une forte
            # probabilité d'avoir true)
        else:
            action = self.getPolicy(state)  # Si non l'agent exploite, l'action optimale selon la politique actuelle
            # est choisit (ceci correspond à une intensification, c'est associé à une faible probabilité d'avoir true)
        return action  # Retour de l'action ou de None si pas daction

    def update(self, state, action, nextState, reward):
        """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
        qSa = self.getQValue(state, action)  # Récupération de la valeur actuelle estimée pour l'état state et l'action.
        sample = reward + self.discount * self.getValue(
            nextState)  # Détermination de la nouvelle estimation de la Q-valeur basée sur l'observation actuelle
        self.qVals[(state, action)] = (1 - self.alpha) * qSa + self.alpha * sample  # Mise à jour de la Q(s,a) selon
        # une moyenne pondérée entre


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        self.weights = util.Counter()  # Initialisation des poids comme un Counter (dictionnaire avec valeur par
        # défaut 0)

    def getQValue(self, state, action):
        """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
        features = self.featExtractor.getFeatures(state, action)  # Récupérer les caractéristiques pour l'état et
        # l'action donnés
        q_value = sum(self.weights[feature] * value for feature, value in features.items())  # Calculer le produit
        # scalaire entre les poids et les caractéristiques
        return q_value

    def update(self, state, action, nextState, reward):
        """
        Updates the weights based on the transition.
        """
        features = self.featExtractor.getFeatures(state, action)  # Récupérer les caractéristiques pour l'état et
        # l'action donnés

        # Calculer l'erreur de différence temporelle (TD)
        q_current = self.getQValue(state, action)
        q_next = self.getValue(nextState)
        td_error = (reward + self.discount * q_next) - q_current

        # Mettre à jour chaque poids
        for feature, value in features.items():
            self.weights[feature] += self.alpha * td_error * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        if self.episodesSoFar == self.numTraining: # Vérifier si la phase d'entraînement est terminée

            print("Final weights:", self.weights) # Afficher les poids pour analyse
