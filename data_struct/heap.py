# -*- coding: utf-8 -*-


class Heap(object):
    """
    # 创建一个最大堆
    """
    heap_list = []
    length = 0
    sorted_list = []

    def __init__(self, heap_list):
        self.heap_list = heap_list[:]  # 这是复制操作, 不是引用, 可以保持原来的数组不变
        self.length = len(heap_list)

    def adjust(self):
        """
        # 从尾部开始调整堆
        # 递归调整, 直到符合堆的定义
        """
        last_index = self.length - 1  # 最后的节点序号
        last_father = (last_index - 1) / 2  # 最后一个父节点序号
        for i in range(last_father + 1):
            now_index = last_father - i  # 从最后一个父节点开始调整
            self.swap(now_index)

    def swap(self, father_index):
        """
        # 具体调整细节, 3个节点找到最大的变成父节点的值
        # 子节点要递归调整
        """
        left_son = father_index * 2 + 1
        right_son = father_index * 2 + 2

        # 找到较大的子节点
        if right_son < self.length:
            if self.heap_list[left_son] > self.heap_list[right_son]:
                max_son = left_son
            else:
                max_son = right_son
        else:
            max_son = left_son

        # 和较大的子节点交换
        if self.heap_list[max_son] > self.heap_list[father_index]:
            self.heap_list[max_son], self.heap_list[father_index] = \
                self.heap_list[father_index], self.heap_list[max_son]  # 交换

        # 传入的节点必须是父节点
        if max_son <= (self.length - 2) / 2:
            self.swap(max_son)

    def get_max_num(self):
        """
        # 调整堆, 取出堆头, 然后用堆尾元素替换被取出的推头元素, 构成新的堆, 长度减去1
        """
        self.adjust()
        max_num = self.heap_list[0]
        self.sorted_list.append(max_num)
        self.heap_list[0] = self.heap_list[self.length - 1]
        self.length -= 1
        return max_num

    def get_sorted_list(self):
        now_length = self.length
        for i in range(now_length):
            self.get_max_num()
        return self.sorted_list


if __name__ == "__main__":
    test_list = [3, 2, 1, 5, 6, 7]
    temp_heap = Heap(test_list)
    print(temp_heap.length)
    temp_heap.adjust()
    print(temp_heap.heap_list)
    temp_heap.get_max_num()
    print(temp_heap.heap_list)
    temp_heap.get_sorted_list()
    print(temp_heap.sorted_list)
    print(test_list)
    pass
