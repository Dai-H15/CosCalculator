import numpy as np


class CosData():
    def __init__(self, name: str, datalist: list[int | float | str]):
        self.name: str = name
        self.datalist: list[int | float | str] = datalist
        self.result: int = 0

    def calc_with(self, *argv: list):
        target: CosData = argv[0]
        vector_A = np.array(self.datalist)
        vector_B = np.array(target.datalist)
        cosine_similarity = np.dot(vector_A, vector_B) / (np.linalg.norm(vector_A) * np.linalg.norm(vector_B))
        self.result = cosine_similarity
        return self.__str__(target=target)

    def __str__(self, target: 'CosData' = None):
        if target is not None:
            return f"{self.name} x {target.name} 's result = {self.result}"
        return f"{self.name} x target 's result = {self.result}"


class CosCalc():
    def __init__(self, target: CosData, data: list[CosData]):
        self.target: CosData = target
        self.data: list[CosData] = data

    def calc_all(self):
        for a in self.data:
            vector_A = np.array(a.datalist)
            vector_B = np.array(self.target.datalist)
            cosine_similarity = np.dot(vector_A, vector_B) / (np.linalg.norm(vector_A) * np.linalg.norm(vector_B))
            a.result = cosine_similarity
        return self.__str__()

    def __str__(self, data: list[CosData] = None):
        res = "\n".join(i.__str__(target=self.target) for i in self.data)
        if data is not None:
            res = "\n".join(i.__str__(target=self.target) for i in data)
        return res

    def max_sort(self):
        check = 0
        check += sum([x.result for x in self.data])
        if check == 0:
            self.calc_all()
        self.sorted_data = sorted(self.data, key=lambda x: x.result, reverse=True)
        print(" ".join(f"{x.name}→" for x in self.sorted_data))
        return self.__str__(self.sorted_data)
