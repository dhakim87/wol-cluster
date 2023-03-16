class SetJoiner:
    def __init__(self, initial_groups):
        self.group_pointers = {g: g for g in initial_groups}

    def top(self, group):
        cur = group
        while self.group_pointers[cur] != cur:
            cur = self.group_pointers[cur]
        return cur

    def merge(self, a, b):
        top_a = self.top(a)
        top_b = self.top(b)
        new_top = top_a if top_a < top_b else top_b
        for cur in [a, b]:
            curs = [cur]
            while self.group_pointers[cur] != cur:
                cur = self.group_pointers[cur]
                curs.append(cur)
            for g in curs:
                self.group_pointers[g] = new_top

    def get_sets(self):
        sets = {}
        for g in self.group_pointers:
            rep = self.top(g)
            if rep not in sets:
                sets[rep] = set()
            sets[rep].add(g)
        return sets


def kruskal_top(groups, i):
    if i not in groups:
        return i
    cur = i
    while groups[cur] != cur:
        cur = groups[cur]
    return cur

