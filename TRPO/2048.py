import random
#import numpy as np

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

BLANK = 0

class Game():
    def __init__(self, n = 4):
        self.n = n
        self.board = [[BLANK for i in range(self.n)] for i in range(self.n)]
        self.score = 0

    def reset(self):
        self.reinit()
        return self.board

    def step(self, action):
        done = self.update(action)
        reward = self.score / 100

        return self.board, reward, done

    def reinit(self):
        for i in range(self.n):
            for j in range(self.n):
                self.board[i][j] = BLANK

        for i in range(2):   
            self.random_piece()

        self.score = 0

    def random_piece(self):
        xx = random.randint(0, self.n - 1)
        yy = random.randint(0, self.n - 1)
        while(self.board[xx][yy] != BLANK):
            xx = random.randint(0, self.n - 1)
            yy = random.randint(0, self.n - 1)
        self.board[xx][yy] = 2

    def update(self, action):
        self.move(action)
        self.merge(action)
        self.move(action)
        self.random_piece()

        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == BLANK:
                    return False
        return True

    def move(self, action):
        if action == UP:
            for i in range(self.n):
                for j in range(self.n):
                    if self.board[i][j] != BLANK:
                        x = i
                        y = j
                        while(self.isvalid(x - 1, y)):
                            x -= 1
                        tmp = self.board[i][j]
                        self.board[i][j] = BLANK
                        self.board[x][y] = tmp

        elif action == DOWN:
            for i in range(self.n - 1, -1, -1):
                for j in range(self.n):
                    if self.board[i][j] != BLANK:
                        x = i
                        y = j
                        while(self.isvalid(x + 1, y)):
                            x += 1
                        tmp = self.board[i][j]
                        self.board[i][j] = BLANK
                        self.board[x][y] = tmp

        elif action == LEFT:
            for i in range(self.n):
                for j in range(self.n):
                    if self.board[i][j] != BLANK:
                        x = i
                        y = j
                        while(self.isvalid(x, y - 1)):
                            y -= 1
                        tmp = self.board[i][j]
                        self.board[i][j] = BLANK
                        self.board[x][y] = tmp

        elif action == RIGHT:
            for i in range(self.n):
                for j in range(self.n - 1, -1, -1):
                    if self.board[i][j] != BLANK:
                        x = i
                        y = j
                        while(self.isvalid(x, y + 1)):
                            y += 1
                        tmp = self.board[i][j]
                        self.board[i][j] = BLANK
                        self.board[x][y] = tmp

    def merge(self, action):
        if action == UP:
            for i in range(self.n):
                for j in range(self.n):
                    if self.isin(i + 1, j) and self.board[i][j] != BLANK and self.board[i + 1][j] == self.board[i][j]:
                        self.board[i][j] *= 2
                        self.board[i + 1][j] = 0
                        self.score += self.board[i][j]

        elif action == DOWN:
            for i in range(self.n - 1, -1, -1):
                for j in range(self.n):
                    if self.isin(i - 1, j) and self.board[i][j] != BLANK and self.board[i - 1][j] == self.board[i][j]:
                        self.board[i][j] *= 2
                        self.board[i - 1][j] = 0
                        self.score += self.board[i][j]

        elif action == LEFT:
            for i in range(self.n):
                for j in range(self.n):
                    if self.isin(i, j + 1) and self.board[i][j] != BLANK and self.board[i][j + 1] == self.board[i][j]:
                        self.board[i][j] *= 2
                        self.board[i][j + 1] = 0
                        self.score += self.board[i][j]

        elif action == RIGHT:
            for i in range(self.n):
                for j in range(self.n - 1, -1, -1):
                    if self.isin(i, j - 1) and self.board[i][j] != BLANK and self.board[i][j - 1] == self.board[i][j]:
                        self.board[i][j] *= 2
                        self.board[i][j - 1] = 0
                        self.score += self.board[i][j]

    def isvalid(self, i, j):
        return (0 <= i < self.n and 0 <= j < self.n and self.board[i][j] == BLANK)

    def isin(self, i, j):
        return (0 <= i < self.n and 0 <= j < self.n)

if __name__ == '__main__':
    game = Game()
    obs = game.reset()
    done = False
    for i in range(game.n):
        for j in range(game.n):
            print obs[i][j] ,
        print
    while(not done):
        #cmd = input()
        cmd = random.randint(0,3)
        print 'cmd: ', cmd
        obs, _, done = game.step(cmd)
        '''
        for i in range(game.n):
            for j in range(game.n):
                print obs[i][j] ,
            print
        '''
    print game.score
