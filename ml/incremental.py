"""Update changed feature blocks between leaves, including sibling/unrelated positions.

No move-history assumptions or search hooks: each block is keyed by all of its
board dependencies. Storage is bounded to the last entry per feature block.
"""
import chess
import numpy as np
from ml.model import (encode_state, RELATIONAL_INPUT_SIZE, _KING_GEOMETRY,
                      _PASSED_MASKS, _SHIELD_MASKS)

class IncrementalEncoder:
    def __init__(self, input_size):
        self.input_size=input_size
        self.values=np.zeros(input_size,dtype=np.float32)
        self.masks=[0]*12
        self.geometry={}
        self.pawns={}
        self.reused_geometry=0

    def encode(self, board):
        """Return a borrowed feature buffer, valid until the next encode call."""
        masks={}
        for ci,color in enumerate((chess.WHITE,chess.BLACK)):
            for pt in chess.PIECE_TYPES:
                index=ci*6+pt-1;mask=board.pieces_mask(pt,color);masks[color,pt]=mask
                old=self.masks[index]
                for sq in chess.scan_forward(old & ~mask):self.values[index*64+sq]=0
                for sq in chess.scan_forward(mask & ~old):self.values[index*64+sq]=1
                self.masks[index]=mask
        encode_state(board,self.values,self.input_size)
        if self.input_size==RELATIONAL_INPUT_SIZE:self.values[794:]=self.relationships(board,masks)
        return self.values

    def relationships(self, board, masks):
        attacks={}
        for color in (chess.WHITE,chess.BLACK):
            combined=0
            for sq in chess.scan_forward(board.occupied_co[color]):
                combined|=board.attacks_mask(sq)
            attacks[color]=combined
        result=[]
        for color in (chess.WHITE,chess.BLACK):
            direction=1 if color else -1
            own,enemy=board.king(color),board.king(not color)
            for pt in range(1,6):
                signature=(masks[color,pt],own if masks[color,pt] else None,enemy if masks[color,pt] else None)
                previous=self.geometry.get((color,pt))
                if previous is not None and previous[0]==signature:
                    block=previous[1];self.reused_geometry+=1
                else:
                    block=[]
                    squares=list(chess.scan_forward(masks[color,pt]))
                    for king in (own,enemy):
                        proximity=forward=lateral=0.
                        if king is not None:
                            for sq in squares:
                                a,b,c=_KING_GEOMETRY[king][sq]
                                proximity+=a;forward+=b;lateral+=c
                        block.extend((proximity/8,direction*forward/8,lateral/8))
                    self.geometry[color,pt]=(signature,block)
                result.extend(block)
            for pt in range(1,6):
                occupied=masks[color,pt]
                attacked=occupied & attacks[not color];defended=occupied & attacks[color]
                result.extend((attacked.bit_count()/8,defended.bit_count()/8,(attacked & ~defended).bit_count()/8))
            signature=(masks[color,chess.PAWN],masks[not color,chess.PAWN],own)
            if not signature[0]:signature=(0,0,None)
            previous=self.pawns.get(color)
            if previous is not None and previous[0]==signature:block=previous[1]
            else:
                pawns=list(chess.scan_forward(signature[0]));files=[sq&7 for sq in pawns];distinct=set(files)
                isolated=sum(f-1 not in distinct and f+1 not in distinct for f in files)
                doubled=len(files)-len(distinct)
                passed=sum(not (signature[1] & _PASSED_MASKS[color][sq]) for sq in pawns)
                shield=(signature[0] & _SHIELD_MASKS[color][own]).bit_count() if own is not None else 0
                block=(isolated/8,doubled/8,passed/8,shield/8);self.pawns[color]=(signature,block)
            result.extend(block)
        return result


class IncrementalBaseline:
    """Reuse exact handcrafted blocks; constants are fixed for the evaluator."""
    def __init__(self):
        import engine.evaluation as hce
        self.hce=hce
        self.pieces=[None]*12
        self.activity=[None]*6
        self.tables={}
        for color in (chess.WHITE,chess.BLACK):
            for pt,table in hce.PST.items():
                oriented=tuple(table[sq^56 if color else sq] for sq in range(64))
                self.tables[color,pt]=oriented
                if pt==chess.KING:
                    for phase in range(hce.MAX_PHASE+1):
                        self.tables[color,pt,phase]=tuple((oriented[sq]*phase+
                            (40-10*(abs(2*(sq&7)-7)+abs(2*(sq>>3)-7)))*(hce.MAX_PHASE-phase))//hce.MAX_PHASE
                            for sq in range(64))

    def evaluate(self,board):
        hce=self.hce
        global_masks=(board.pawns,board.knights,board.bishops,board.rooks,board.queens,board.kings)
        masks=[tuple(mask & board.occupied_co[color] for mask in global_masks) for color in (chess.WHITE,chess.BLACK)]
        phase=min(hce.MAX_PHASE,sum(weight*(global_masks[pt-1] & board.occupied).bit_count()
                                   for pt,weight in hce.PHASE_WEIGHTS.items()))
        total=0
        for ci,color in enumerate((chess.WHITE,chess.BLACK)):
            score=0;own=masks[ci];enemy=masks[1-ci]
            for pi,mask in enumerate(own):
                index=ci*6+pi;pt=pi+1;king_phase=phase if pt==chess.KING and mask else -1
                previous=self.pieces[index]
                if previous is None or previous[0]!=mask or previous[1]!=king_phase:
                    table=self.tables[color,pt,phase] if pt==chess.KING else self.tables[color,pt]
                    value=mask.bit_count()*hce.MATERIAL[pt]+sum(table[sq] for sq in chess.scan_forward(mask))
                    previous=(mask,king_phase,value);self.pieces[index]=previous
                score+=previous[2]
            score+=hce._mobility_score(board,color)+hce._center_control_score(board,color)
            minors=own[1]|own[2];key=(minors,own[4],phase)
            previous=self.activity[ci*3]
            if previous is None or previous[0]!=key:
                home=chess.BB_RANK_1 if color else chess.BB_RANK_8
                undeveloped=(minors & home).bit_count()
                development=(minors.bit_count()-undeveloped)*hce.DEVELOPMENT_BONUS
                if own[4] and not own[4] & chess.BB_SQUARES[chess.D1 if color else chess.D8] and undeveloped>=2:
                    development-=hce.QUEEN_EARLY_DEVELOPMENT_PENALTY
                previous=(key,development*phase//hce.MAX_PHASE);self.activity[ci*3]=previous
            score+=previous[1]
            for offset,key,fn in [(1,(own[3],own[0],enemy[0]),hce._rook_activity_score),
                                  (2,(own[0],enemy[0]),hce._passed_pawn_score)]:
                previous=self.activity[ci*3+offset]
                if previous is None or previous[0]!=key:
                    previous=(key,fn(board,color));self.activity[ci*3+offset]=previous
                score+=previous[1]
            total+=(1 if color else -1)*score
        return total
