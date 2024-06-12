#include "ukernels.h"

#ifdef OMP_ENABLE
  #include <omp.h>
#endif

void im2row(AB_TYPE *rows, int ld, AB_TYPE *in, int batch, 
	    int height, int width, int channel, int oheight, 
	    int owidth, int kheight, int kwidth, 
	    int vpadding, int hpadding, int vstride, int hstride, 
	    int vdilation, int hdilation, int TH);
