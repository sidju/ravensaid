#pragma once

typedef struct _RavensaidState RavensaidState;

// Takes path to saved neural network
// Returns pointer if successful
// Returns NULL if reading fails
RavensaidState* ravensaid_init(char*);

// Takes pointer and frees the memory
// If given NULL will halt program execution
void ravensaid_free(RavensaidState*);

// Takes state of a loaded neural network and a message to rate
// Returns percentage probability of message being written by 
// Ravenholdt as a fixed precision int (2 decimal places)
// Returns -1 if message is invalid (bad length or NULL)
// Returns -2 if the probability was over 200%
// Returns -3 if the probability was negative
int ravensaid(RavensaidState*, char*);
