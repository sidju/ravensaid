// Linking with the libravensaid.a binary requires having libtorch installed
// (This is sometimes its own package but often part of the pytorch package)

#include "ravensaid.h"
#include <stdio.h>

int main(int argc, char** argv) {
  RavensaidState* state = ravensaid_init("loss_0.4625_77.33_percent.nn");
  if (state == 0){
    printf("Error, couldn't load neural network from file.\n");
    return 1;
  }

  char* message = "...";
  int probability = ravensaid(state, message);
  if (probability < 0) {
    printf("Error, failed processing message.\n");
    return 2;
  }
  printf(
    "Message: \"%s\"\nProbability of being written by Ravenholdt: %d,%d%%\n",
    message,
    probability / 100,
    probability % 100
  );

  ravensaid_free(state);
  return 0;
}
