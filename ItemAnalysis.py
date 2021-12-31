import numpy as np
import copy

''' Minimum distance threshold '''
item_min_dist = 0.1


def distance(a, b):
    '''
    Get the distance between a and b
    '''
    dist = np.linalg.norm(a - b)
    return dist


class ItemAnalysis:
    def __init__(self):
        ''' ALl vehicle items '''
        self.items = {}

        ''' ALl previous vehicle items '''
        self.pre_items = {}

        ''' Difference between previous and current items' positions  '''
        self.vectors = {}

        ''' Previous rotations of vehicles' items '''
        self.rots = {}

    def step(self):
        ''' Make a copy of items '''
        self.pre_items = copy.deepcopy(self.items)

        ''' Remove all items in the current item list '''
        self.items.clear()

    def prepare_vectors(self):
        ''' Create same size list of items by making copy of items '''
        self.vectors = copy.deepcopy(self.items)

        for i in self.items.keys():
            if i in self.pre_items:
                ''' X position of the vector '''
                self.vectors[i][0] = self.items[i][0] - self.pre_items[i][0] if (self.items[i][0] - self.pre_items[i][
                    0]) < 100000 else 100000
                ''' Y position of the vector '''
                self.vectors[i][1] = self.items[i][1] - self.pre_items[i][1] if (self.items[i][1] - self.pre_items[i][
                    1]) < 100000 else 100000

    def add_item(self, id, x, y):
        ''' Add a new item to items' list '''
        item = [x, y]
        self.items[id] = item

    def get_rot_dist(self, id):
        ''' Initiate the vector list '''
        self.prepare_vectors()

        ''' If id has not been added to item list yet, return 0 '''
        if not id in self.items.keys():
            return [0, 0]

        ''' If id was inserted in the previous items' list'''
        if id in self.pre_items.keys():
            item = np.float32(self.vectors[id])

            ''' Calc the angle '''
            ang = np.arctan2(item[1], item[0])

            ''' Calc the distance '''
            dist = np.linalg.norm(item)

            ''' If distance between previous and current position is more than 1 '''
            if dist > 5:
                ''' If id has been added to rots before '''
                if id in self.rots:
                    ''' if angle difference is not a lot ignore it '''
                    # if abs(ang - self.rots[id]) > 0.3:
                    self.rots[id] = ang
                    return [self.rots[id], dist]
            else:
                ''' If id has been added to rots before '''
                if id in self.rots:
                    return [self.rots[id], dist]
                else:
                    self.rots[id] = ang
                    return [self.rots[id], dist]

            return [ang, dist]
        return [0, 0]
