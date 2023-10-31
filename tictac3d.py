"""
Basic 3D Tic Tac Toe with Minimax and Alpha-Beta pruning, using a simple
heuristic to check for possible winning moves or blocking moves if no better
alternative exists.
"""


from shutil import move
# from colorama import Back, Style, Fore
import numpy as np

class TicTacToe3D(object):
    """3D TTT logic and underlying game state object.

    Attributes:
        board (np.ndarray)3D array for board state.
        difficulty (int): Ply; number of moves to look ahead.
        depth_count (int): Used in conjunction with ply to control depth.

    Args:
        player (str): Player that makes the first move.
        player_1 (Optional[str]): player_1's character.
        player_2 (Optional[str]): player_2's character.
        ply (Optional[int]): Number of moves to look ahead.
    """

   

    def __init__(self, board = None, player=-1, player_1=-1, player_2=1, ply=3):
        if board is not None:
            assert type(board) == np.ndarray, "Board must be a numpy array"
            assert board.shape == (3,3,3), "Board must be 3x3x3"
            self.np_board = board
        else:
            self.np_board = self.create_board()
        self.map_seq_to_ind, self.map_ind_to_seq = self.create_map()
        self.allowed_moves = list(range(pow(3, 3)))
        self.difficulty = ply
        self.depth_count = 0
        if player == player_1:
            self.player_1_turn = True
        else:
            self.player_1_turn = False
        self.player_1 = player_1
        self.player_2 = player_2
        self.players = (self.player_1, self.player_2)

    def create_map(self):
        """Create a mapping between index of 3D array and list of sequence, and vice-versa.

        Args: None

        Returns:
            map_seq_to_ind (dict): Mapping between sequence and index.
            map_ind_to_seq (dict): Mapping between index and sequence.
        """
        a = np.hstack((np.zeros(9),np.ones(9),np.ones(9)*2))
        a = a.astype(int)
        b = np.hstack((np.zeros(3),np.ones(3),np.ones(3)*2))
        b = np.hstack((b,b,b))
        b = b.astype(int)
        c = np.array([0,1,2],dtype=int)
        c = np.tile(c,9)
        mat = np.transpose(np.vstack((a,b,c)))
        ind = np.linspace(0,26,27).astype(int)
        map_seq_to_ind = {}
        map_ind_to_seq = {}
        for i in ind:
            map_seq_to_ind[i] = (mat[i][0],mat[i][1],mat[i][2])
            map_ind_to_seq[(mat[i][0],mat[i][1],mat[i][2])] = i
        return map_seq_to_ind, map_ind_to_seq
    
    def reset(self):
        """Reset the game state."""
        self.allowed_moves = list(range(pow(3, 3)))
        self.np_board = self.create_board()
        self.depth_count = 0

    def check_board(self):
        ind = np.linspace(0,26,27).astype(int)
        for i in ind:
            if(self.np_board[self.map_seq_to_ind[i]] !=0):
                self.allowed_moves.remove(i)
                self.allowed_moves.sort()



    @staticmethod
    def create_board():
        """Create the board with appropriate positions and the like

        Returns:
            np_board (numpy.ndarray):3D array with zeros for each position.
        """
        np_board = np.zeros((3,3,3), dtype=int)
        return np_board

   # all possible winning moves on the board
    winning_combos = (
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14],
        [15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26],

        [0, 3, 6], [1, 4, 7], [2, 5, 8], [9, 12, 15], 
        [10, 13, 16], [11, 14, 17], [18, 21, 24], [19, 22, 25], 
        [20, 23, 26],

        [0, 4, 8], [2, 4, 6], [9, 13, 17], [11, 13, 15], 
        [18, 22, 26], [20, 22, 24],

        [0, 9, 18], [1, 10, 19], [2, 11, 20], [3, 12, 21], 
        [4, 13, 22], [5, 14, 23], [6, 15, 24], [7, 16, 25], 
        [8, 17, 26],

        [0, 12, 24], [6, 12, 18], [2, 14, 26], [8, 14, 20],  
        [0, 10, 20], [2, 10, 18], [6, 16, 26], [8, 16, 24], 
        [1, 13, 25], [7, 13, 19], [3, 13, 23], [5, 13, 21],  
        [0, 13, 26], [2, 13, 24], [6, 13, 20], [8, 13, 18]
    )
   
    @property
    def game_over(self):
        """Checking if the game is over or not by going through 
        the board to see if any of the winning moves have been played"""

        win= False
        for player in self.players:
            for i in self.winning_combos:
                count=0
                for j in i:
                    if(self.np_board[self.map_seq_to_ind[j]] == player):
                        count+=1
                        if(count == 3):
                            win = True
                            return win
                    else:
                        win = False
                        break
        return win

    @property
    def get_winner(self):
        """Returns who the winner is."""
        for player in self.players: 
            for i in self.winning_combos:
                count = 0
                for j in i:
                    if(self.np_board[self.map_seq_to_ind[j]]== player):
                        count+=1
                        if count == 3:
                            return player
        return None                   

    
    def make_move(self, player, location):
        """Function to make a move on the board"""
        self.allowed_moves.remove(location)
        self.allowed_moves.sort()
        self.np_board[self.map_seq_to_ind[location]] = player
    
    def undo_move(self, location):
        """Function to undo a move on the board"""
        self.allowed_moves.append(location)
        self.allowed_moves.sort()
        self.np_board[self.map_seq_to_ind[location]] = 0
    
    
    def opp_player(self, player):
        """Function to get the opposite player when the input is 
        the current player"""

        if(player == -1):
            return 1
        else:
            return -1

    
    def available_winning_moves(self, player):
        """Returns the number of potential winning moves that can be 
        made by a particular player.
        
        Checks how many 1 in a rows, 2 in a rows and 3 in a rows combos
        each player has and assigns a weight for each.
        Player 1 has slightly more weightage so it blocks player 2. """
        
            
        opp_player = self.opp_player(player)
        win_moves = 0
        for combo in self.winning_combos:
            if player == -1:
                if all(self.np_board[self.map_seq_to_ind[j]] != opp_player for j in combo):
                    count = 0
                    for i in combo:
                        if self.np_board[self.map_seq_to_ind[i]] == player:
                            count+=1
                    if count == 0:
                        win_moves+=1  
                    elif count == 1:
                        win_moves+=2
                    elif count == 2:
                        win_moves+=100
                    elif count == 3:
                        win_moves+=4901
                      
            elif player == 1:
                if all(self.np_board[self.map_seq_to_ind[j]] != opp_player for j in combo):
                    count = 0
                    for i in combo:
                        if i == player:
                            count+=1
                    if count == 0:
                        win_moves+=1  
                    elif count == 1:
                        win_moves+=1.5
                    elif count == 2:
                        win_moves+=80
                    elif count == 3:
                        win_moves+=4000
                           
        return win_moves        

    
    def eval(self):
        """An evaluator function that returns the difference between the available winning moves
        for both players.
        Since player 1 is max: higher the eval value better it is for player 1"""

        winning_moves_player1 = self.available_winning_moves(self.player_1)
        winning_moves_player2 = self.available_winning_moves(self.player_2)
        return winning_moves_player1 - winning_moves_player2

    def max_game(self, player, alpha, beta):
        """The max function for player 1's best move."""

        if self.depth_count == self.difficulty:
            return self.eval()-(self.depth_count*50)  
        if(self.depth_count <=self.difficulty):
            self.depth_count+=1
            v = -5000
            for move in self.allowed_moves:
                self.make_move(player, move)
                if self.game_over:
                    v = self.eval()
                    self.undo_move(move)
                    return v
                else:
                    v = self.min_game(self.opp_player(player), alpha,beta)
                    if v > alpha:
                        alpha = v
                        self.undo_move(move)
                    else:    
                        self.undo_move(move) 
                
                if alpha >= beta:
                    break
            return alpha
        
        else:
            return self.eval()



    def min_game(self, player, alpha, beta):
        """The min function for player 2's best move"""

        if self.depth_count == self.difficulty:
            return self.eval()-(self.depth_count*50)
        if(self.depth_count <=self.difficulty):
            self.depth_count += 1
            v = 5000
            
            for move in self.allowed_moves:
                self.make_move(player, move)
                if self.game_over:
                    v = self.eval()
                    self.undo_move(move)
                    
                    return v
                else:
                    v = self.max_game(self.opp_player(player), alpha,beta)
                    if v < beta:
                        beta = v
                        self.undo_move(move)
                    else:    
                        self.undo_move(move) 
                
                if alpha >= beta:
                    break
            return beta
        
        else:
            return self.eval()

    def player_1_move(self):
        """Player 1's turn to make a move"""
        max_eval = -5000
        win = False
        for move in self.allowed_moves:
            self.make_move(self.player_1, move)
            if self.game_over:
                win = True
                break
            else:
                v = self.min_game(self.player_2, -5000, 5000)
                self.depth_count = 0
                if v>max_eval:
                    max_eval = v
                    best_move = move
                    self.undo_move(move)
                else:
                    self.undo_move(move)  

                #checking if this move will block the opponent from making the winning move
                self.make_move(self.player_2, move)
                #if the opponent makes the winning move and game is over-
                if self.game_over and self.get_winner == self.player_2:
                    #if max eval is still lesser than the best score for player 1
                    if  max_eval <= 5001:
                        max_eval = 5001
                        best_move = move
                self.undo_move(move)  

        if not win:
            self.make_move(self.player_1, best_move)
        self.player_1_turn = False
               
    def player_2_move(self):
        """Player 2's turn to make a move"""
        min_eval = 5000
        win = False
        for move in self.allowed_moves:
            self.make_move(self.player_2, move)
            if self.game_over:
                win = True
                break
            else:
                v = self.max_game(self.player_1, -5000, 5000)
                self.depth_count = 0
                if v<min_eval:
                    min_eval = v
                    best_move = move
                    self.undo_move(move)
                else:
                    self.undo_move(move)  

                #checking if this move will block the opponent from making the winning move
                self.make_move(self.player_1, move)
                #if the opponent makes the winning move and game is over-
                if self.game_over and self.get_winner == self.player_1:
                    #if min eval is still greater than the best score for player 2      
                    if min_eval >= -5001:                                    
                        min_eval = -5001
                        best_move = move
                self.undo_move(move)  

        if not win:
            self.make_move(self.player_2, best_move)
        self.player_1_turn = True  
                     

    def play_game(self):
        """Primary game loop.

        Until the game is complete we will alternate between computer and
        player turns while printing the current game state.
        """
        self.check_board()
       
        try:
            while not self.game_over:
                    if self.player_1_turn:
                        self.player_1_move()
                        
                    else:
                        self.player_2_move()
                        
            print("final board: \n{}".format(self.np_board))
            return self.np_board, self.get_winner
        except KeyboardInterrupt:
            print('\n ctrl-c detected, exiting')



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--player', dest='player', help='Player that plays first, 1 or -1',\
                type=int, default=-1, choices=[1,-1]
    )
    parser.add_argument(
        '--ply', dest='ply', help='Number of moves to look ahead', \
                type=int, default=6
    )
    args = parser.parse_args()
    brd,winner = TicTacToe3D(player=args.player, ply=args.ply).play_game()
    print("final board: \n{}".format(brd))
    print("winner: player {}".format(winner))
