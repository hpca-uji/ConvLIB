#include "ukernels.h"

void im2col(AB_TYPE *restrict cols, int ld, const AB_TYPE *restrict in, 
	    int batches, int channels, int height, int width,
            int oheight, int owidth, int kheight, int kwidth, 
	    int vpadding, int hpadding, int vstride, int hstride,
            int vdilation, int hdilation);
