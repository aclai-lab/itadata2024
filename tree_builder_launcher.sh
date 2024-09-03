#!/bin/bash

N_THREADS=30
# tmux a -t Session_1
# tmux kill-session -t Session_1

# ------------------------------------------------------------------------------------ #
#                                       Pneumonia                                      #
# ------------------------------------------------------------------------------------ #
# EXPERIMENT="Pneumonia"
# SCALE="semitones"
# USE_MFCC="false"
# n=1
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="Pneumonia"
# SCALE="semitones"
# USE_MFCC="true"
# n=2
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="Pneumonia"
# SCALE="mel_htk"
# USE_MFCC="false"
# n=3
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="Pneumonia"
# SCALE="mel_htk"
# USE_MFCC="true"
# n=4
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                    Bronchiectasis                                    #
# ------------------------------------------------------------------------------------ #
# EXPERIMENT="Bronchiectasis"
# SCALE="semitones"
# USE_MFCC="false"
# n=1
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="Bronchiectasis"
# SCALE="semitones"
# USE_MFCC="true"
# n=2
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="Bronchiectasis"
# SCALE="mel_htk"
# USE_MFCC="false"
# n=3
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="Bronchiectasis"
# SCALE="mel_htk"
# USE_MFCC="true"
# n=4
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                          COPD                                        #
# ------------------------------------------------------------------------------------ #
# EXPERIMENT="COPD"
# SCALE="semitones"
# USE_MFCC="false"
# n=1
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="COPD"
# SCALE="semitones"
# USE_MFCC="true"
# n=2
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="COPD"
# SCALE="mel_htk"
# USE_MFCC="false"
# n=3
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="COPD"
# SCALE="mel_htk"
# USE_MFCC="true"
# n=4
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                          URTI                                        #
# ------------------------------------------------------------------------------------ #
# EXPERIMENT="URTI"
# SCALE="semitones"
# USE_MFCC="false"
# n=1
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="URTI"
# SCALE="semitones"
# USE_MFCC="true"
# n=2
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="URTI"
# SCALE="mel_htk"
# USE_MFCC="false"
# n=3
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# EXPERIMENT="URTI"
# SCALE="mel_htk"
# USE_MFCC="true"
# n=4
# session="Session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                     Bronchiolitis                                    #
# ------------------------------------------------------------------------------------ #
EXPERIMENT="Bronchiolitis"
SCALE="semitones"
USE_MFCC="false"
n=1
session="Session_"$n
tmux new-session -d -s $session
tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
[ -n "${TMUX:-}" ]

EXPERIMENT="Bronchiolitis"
SCALE="semitones"
USE_MFCC="true"
n=2
session="Session_"$n
tmux new-session -d -s $session
tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
[ -n "${TMUX:-}" ]

EXPERIMENT="Bronchiolitis"
SCALE="mel_htk"
USE_MFCC="false"
n=3
session="Session_"$n
tmux new-session -d -s $session
tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
[ -n "${TMUX:-}" ]

EXPERIMENT="Bronchiolitis"
SCALE="mel_htk"
USE_MFCC="true"
n=4
session="Session_"$n
tmux new-session -d -s $session
tmux send-keys -t $session 'julia -i -t '$N_THREADS' modal_trees_builder.jl '$EXPERIMENT' '$SCALE' '$USE_MFCC' &' Enter
[ -n "${TMUX:-}" ]