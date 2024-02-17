# Brandubh

## Description
A Python implementation of "Brandubh", a small variant of Norse Tafl games that became the preferred version in Ireland. The rules used here are simplified and included below.

This project is the spiritual successor of a previous attempt to code a full version of Copenhagen Hnefatafl and train a superhuman RL agent to play the game. That attempt was archived due to issues with training the agent.

![Game Logic](https://img.shields.io/badge/Game_Logic-Complete-green)
![MCTS](https://img.shields.io/badge/MCTS-Complete-green)
![Deep RL Agent](https://img.shields.io/badge/Deep_RL_Agent-In_Progress-yellow)
![Human Playable](https://img.shields.io/badge/Human_Playable-Not_Started-red)

## Installation
TBD. This project is in the phase of developing the RL agent and is not yet playable by humans. 

## Usage
TBD.

## Contributing
This is currently a solo project, but suggestions or pertinent pull requests are welcome.

## License
Apache 2.0

## Rules
This implementation uses the following ruleset:
* The game is played on a 7x7 board
* The pieces include one King and four defenders (one player), and eight attackers (the other player). The King starts in the middle surrounded by the 4 defenders, The 8 attackers fill the remaining spaces between the king and the edges in each of the 4 cardinal directions.
* Attackers move first.
* Pieces move "like rooks in Chess."
* Any piece may enter the center tile, but only the King may enter a corner (this contradicts some others rules for this game which only allow the King to enter the center tile).
* Pieces are captured by being flanked between two opponents or between an opponent and corner (this contradicts some other rules which allow the center tile to be used in flanking and/or which have special rules for capturing the King).
* The Defenders' side wins when the King reaches a corner. The Attackers' side wins when the King is captured. Either side may win if the opponent has no legal moves.
* There are no stalemates allowed for in this ruleset. Never give up! Never surrender!


##
![vikings_playing_tafl](https://github.com/A-J-V/Brandubh/assets/72227828/0f5cbe6d-5f56-423d-9672-073df531e0c0)
