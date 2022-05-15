"""
refer to
https://github.com/BobscHuang/Gomoku/blob/master/Gomoku/Gomoku.py
"""

from tkinter import Tk, Canvas, Button
from gym import Wrapper

class GUI_Wrapper(Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self._init_setup()
        #Create Board


    def create_circle(self, x, y, radius, fill = "", outline = "black", width = 1):
        self._canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill = fill, outline = outline, width = width)

    def MouseClick(self, event):
        X_click = event.x
        Y_click = event.y
        self.Click_Cord = self.Piece_Location(X_click, Y_click)
        # print(self.Click_Cord)

    def Piece_Location(self, X_click, Y_click):    
        X = None
        Y = None
        for i in range(len(self.Actual_CordX1)):
            
            if X_click > self.Actual_CordX1[i] and X_click < self.Actual_CordX2[i]:
                X = self.Game_CordX[i]

            if Y_click > self.Actual_CordY1[i] and Y_click < self.Actual_CordY2[i]:
                Y = self.Game_CordY[i]

        return X, Y
    def Location_Validation(self):
        if self.X == None or self.Y == None:
            return False
            
        elif self.board[self.Y - 1][self.X - 1] == 0:
            return True
    def Score_Board(self):
        if self.Winner == None:
            Turn_Text = self._canvas.create_text(self.width / 2, self.height - self.Frame_Gap + 15, text = "Turn = " + self.Turn, font = "Helvetica 25 bold", fill = self.Turn)
            return Turn_Text
        else:
            self._canvas.create_text(self.width / 2, self.height - self.Frame_Gap + 15, text = self.Winner.upper() + " WINS!", font = "Helvetica 25 bold", fill = self.Winner.lower())

    

    def Exit(self):
        self.Winner = "Exit"
        self._tk.destroy()

    def _init_setup(self):

        self._tk = Tk()
        self._tk.title(string='Gomoku 五子棋 : MAEG3080 Course Project, Cheers!!')
        self._canvas = Canvas(self._tk, width=800, height=800, background= "#b69b4c")
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self.MouseClick)

        #Board Size
        # self.Board_Size = 15
        self.Board_Size = self.env.width

        self.Frame_Gap = 35
        self.width = 800
        self.height = 800

        #Board
        self.Board_Size = self.Board_Size - 1
        self.Board_X1 = self.width / 10
        self.Board_Y1 = self.height / 10
        self.Board_GapX = (self.width - self.Board_X1 * 2) / self.Board_Size
        self.Board_GapY = (self.height - self.Board_Y1 * 2) / self.Board_Size

        #Chess Piece
        self.Chess_Radius = (self.Board_GapX * (9 / 10)) / 2

        #Turn
        self.Turn = "white"
        self.Winner = None
        self.done =  False

        self.Click_Cord = [None, None]
        #Cord List
        self.Black_Cord_PickedX = []
        self.Black_Cord_PickedY = []
        self.White_Cord_PickedX = []
        self.White_Cord_PickedY = []

        #Click Detection Cord
        self.Game_CordX = []
        self.Game_CordY = []
        self.Actual_CordX1 = []
        self.Actual_CordY1 = []
        self.Actual_CordX2 = []
        self.Actual_CordY2 = []

        #2D Board List
        self.board = []

        #Buttons
        self._button = Button(self._tk, text = "EXIT", font = "Helvetica 10 bold", command = self.Exit, bg = "gray", fg = "black")
        self._button.pack()
        self._button.place(x = self.width / 2 * 0.5, y = self.height - self.Frame_Gap * 1.6 + 15, 
                            height = self.Chess_Radius * 2, width = self.Chess_Radius * 4)

        #2D list for gameboard
        for i in range(self.Board_Size + 1):
            self.board.append([0] * (self.Board_Size + 1))
        
        self.Unfilled = 0
        self.Black_Piece = 1
        self.White_Piece = 2

        #Fills Empty List
        for z in range(1, self.Board_Size + 2):
            
            for i in range(1, self.Board_Size + 2):
                self.Game_CordX.append(z)
                self.Game_CordY.append(i)
                self.Actual_CordX1.append((z - 1) * self.Board_GapX + self.Board_X1 - self.Chess_Radius)
                self.Actual_CordY1.append((i - 1) * self.Board_GapY + self.Board_Y1 - self.Chess_Radius)
                self.Actual_CordX2.append((z - 1) * self.Board_GapX + self.Board_X1 + self.Chess_Radius)
                self.Actual_CordY2.append((i - 1) * self.Board_GapY + self.Board_Y1 + self.Chess_Radius)
        #Create Board
        self._canvas.create_rectangle(self.Board_X1 - self.Frame_Gap, 
                                    self.Board_Y1 - self.Frame_Gap, 
                                    self.Board_X1 + self.Frame_Gap + self.Board_GapX * self.Board_Size, 
                                    self.Board_Y1 + self.Frame_Gap + self.Board_GapY * self.Board_Size, width = 3)

        for f in range(self.Board_Size + 1):
            self._canvas.create_line(self.Board_X1, self.Board_Y1 + f * self.Board_GapY,
                                     self.Board_X1 + self.Board_GapX * self.Board_Size, self.Board_Y1 + f * self.Board_GapY)
            self._canvas.create_line(self.Board_X1 + f * self.Board_GapX, 
                                        self.Board_Y1, self.Board_X1 + f * self.Board_GapX, self.Board_Y1 + self.Board_GapY * self.Board_Size)

            self._canvas.create_text(self.Board_X1 - self.Frame_Gap * 1.7, self.Board_Y1 + f * self.Board_GapY, 
                                        text = f + 1, font = "Helvetica 10 bold", fill = "black")
            self._canvas.create_text(self.Board_X1 + f * self.Board_GapX, self.Board_Y1 - self.Frame_Gap * 1.7, text = f + 1, 
                                        font = "Helvetica 10 bold", fill = "black")

        self.Turn_Text = self.Score_Board()
    
    def run(self, p1=None, p2=None):
        done = False
        while not done:
            self._canvas.update()
            is_action = False
            if p1:
                if self.env.board.get_current_player() % 2 == 1:
                    act = p1.get_action(self.env.board)
                    is_action  = True
                    print("cases", (act+1)%(self.Board_Size+1))
                    if int((act+1)%(self.Board_Size+1)) == 0:
                        print("special")
                        self.X = self.Board_Size+1
                        self.Y = (act+1)//(self.Board_Size+1) 
                    else:
                        self.X = (act+1)%(self.Board_Size+1) 
                        self.Y = (act+1)//(self.Board_Size+1) + 1
                    print("oponent act", act)
            if p2:
                if self.env.board.get_current_player() % 2 == 0:
                    act = p2.get_action(self.env.board)
                    is_action  = True
                    if int((act+1)%(self.Board_Size+1)) == 0:
                        print("special")
                        self.X = self.Board_Size+1
                        self.Y = (act+1)//(self.Board_Size+1) 
                    else:
                        self.X = (act+1)%(self.Board_Size+1) 
                        self.Y = (act+1)//(self.Board_Size+1) + 1
                    # print("oponent act", act)
            
            if not is_action:
                self.X = self.Click_Cord[0]
                self.Y = self.Click_Cord[1]
                Picked = self.Location_Validation()
                if Picked:
                    is_action=True

            if is_action:
                print("==================")
                print(f'Coordinate Y:{self.Y} X:{self.X}' )
                self._canvas.delete(self.Turn_Text)
                
                self.create_circle(self.Board_X1 + self.Board_GapX * (self.X - 1), 
                                    self.Board_Y1 + self.Board_GapY * (self.Y - 1), 
                                    radius = self.Chess_Radius, fill = self.Turn)

                if self.env.board.get_current_player() % 2 == 1:
                    self.White_Cord_PickedX.append(self.X)
                    self.White_Cord_PickedY.append(self.Y)
                    self.board[self.Y - 1][self.X - 1] = 2
                    self.Turn = "black"
                    print("Round side: black")

                elif self.env.board.get_current_player() % 2 == 0:
                    self.Black_Cord_PickedX.append(self.X)
                    self.Black_Cord_PickedY.append(self.Y)
                    self.board[self.Y - 1][self.X - 1] = 1
                    self.Turn = "white"
                    print("Round side: white")

                self.Turn_Text = self.Score_Board()

                action = (self.Y-1) * (self.Board_Size+1) + self.X -1
                print(f'Encoded action: {action}')
                obs, reward, done, info = self.env.step(action)
                # self.env.render()
                print(f'Game end: {done}')
                print("")
                # if self.Turn == "white":
                #     Colour_Check = self.Black_Piece
                #     Win_Check = "Black"

                # elif self.Turn == "black":
                #     Colour_Check = self.White_Piece
                #     Win_Check = "White"


