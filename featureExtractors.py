# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"Feature extractors for Pacman game states"

from game import Directions, Actions
import util


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    - prioritizes eating edible ghosts
    """

    def getFeatures(self, state, action):
        # Extract the grid of food and wall locations, and get the ghost positions and states
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()  # Get GhostState objects (contains positions and scaredTimer)

        features = util.Counter()

        features["bias"] = 1.0

        # Compute the location of PacMan after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(ghost.getPosition(), walls) for ghost in ghosts)

        # Prioritize edible ghosts
        edibleGhostDistances = []
        for ghost in ghosts:
            if ghost.scaredTimer > 0:  # Ghost is edible (scared)
                edibleGhostDistances.append(util.manhattanDistance((next_x, next_y), ghost.getPosition()))

        # If there are any edible ghosts in the vicinity, prioritize them
        if edibleGhostDistances:
            closestEdibleDistance = min(edibleGhostDistances)
            features["eats-edible-ghost"] = 1.0 / (closestEdibleDistance + 1)
            # Bonus to make this much more important than food
            features["eats-edible-ghost"] *= 10.0
        else:
            features["eats-edible-ghost"] = 0.0

        # If PacMan is not near any ghost and there is food to be eaten, add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # Calculate the distance to the closest food
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # Normalize the distance to be a smaller value
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # Normalize the features to prevent overly large values
        features.divideAll(10.0)

        return features


