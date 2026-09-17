/* simple_forward.c – minimal C API usage example */
#include <stdio.h>
#include <stdlib.h>
#include "../Sources/Mamba/MambaConfig.hpp"   /* for layout only – real header is C-compatible */

/* In a real build the C API is exposed via a pure-C header.
   This file shows the intended call sequence. */

int main(void) {
    /*
    MambaConfig cfg = {
        .d_model = 256,
        .d_state = 16,
        .d_conv  = 4,
        .expand  = 2,
        .dtype   = MAMBA_F32
    };

    MambaModel *model = mamba_create(&cfg);
    if (!model) { fprintf(stderr, "create failed\n"); return 1; }

    // load weights ...
    // mamba_load_weights(model, blob, size);

    const int B = 1, L = 128, D = 256;
    float *input  = calloc(B*L*D, sizeof(float));
    float *output = calloc(B*L*D, sizeof(float));

    mamba_forward(model, input, output, B, L);

    // incremental
    MambaStateHandle *st = mamba_create_state(model, B);
    for (int t = 0; t < L; ++t)
        mamba_step(model, st, input + t*D, output + t*D);

    mamba_destroy_state(st);
    mamba_destroy(model);
    free(input); free(output);
    */
    printf("mamba-metal example – see source comments for API usage\n");
    return 0;
}
