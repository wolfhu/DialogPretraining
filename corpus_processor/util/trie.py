# encoding: utf-8

"""
Implementation of Trie.
"""


class Trie:

    END = '##'

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.lookup = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        tree = self.lookup
        for a in word:
            if a not in tree:
                tree[a] = {}
            tree = tree[a]
        # 单词结束标志
        tree[self.END] = self.END

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        tree = self.lookup
        for a in word:
            if a not in tree:
                return False
            tree = tree[a]
        if self.END in tree:
            return True
        return False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        tree = self.lookup
        for a in prefix:
            if a not in tree:
                return False
            tree = tree[a]
        return True

    def sub_tree_leaves(self, prefix):
        tree = self.lookup
        recall_res = []
        for a in prefix:
            if a not in tree:
                return None
            tree = tree[a]

        def dfs(root, prefix):
            if not root:
                return
            if self.END in root:
                recall_res.append(prefix)
            for ch in root:
                if ch == self.END: continue
                dfs(root[ch], prefix + ch)

        dfs(tree, prefix)
        return recall_res
