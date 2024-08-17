
#include "energy.h"

//Energy Consumption
int pmblib_get_data(counter_t pm_counter,line_t lines, int set, double *measures, const int nmeasures) {

/*! Print the structure data in a file in text format.
  The lines and sets parameters define which data will be printed.
  If parameter set is 0 all sets will be printed.
  The format of the file is:
  	Set_id <tab> Time <tab> Value_Line1 <tab> Value_Line2 <tab>... <tab> Value_aggregate

*/

  FILE    *file_data;
  int	i, j, ii, s, init, last, m;
  int	ini, fin, watts_size, interval;
  double	time, inc_time, t, sum;
  int	*ind_print, *ind_lines, n_lines_print, n_lines_counter;

  int n_reads = 0;
	
  for (m=0; m <= nmeasures; m++)
    measures[m] += 0.0; 

  if ( pm_counter.aggregate ) {	//Only aggregate value will be printed
    if (set > pm_counter.measures->energy.watts_sets_size-1 || set <-1)
      return -1;

    if (set == -1) {
      init= 0;
      last= pm_counter.measures->energy.watts_sets_size-1;
    } else {
      init= set;
      last= set+1;
    }

    for( s= init; s < last; s++ ) {
      ini=pm_counter.measures->energy.watts_sets[s];
      fin=pm_counter.measures->energy.watts_sets[s+1];

      watts_size=pm_counter.measures->energy.watts_size;
      time=pm_counter.measures->timing[(s*2)+1]-pm_counter.measures->timing[s*2];
      inc_time=time/(fin-ini-1);

      t=0.0;
      for(i=ini; i<fin; i++){
        printf("%d\t%f\t%f\n", s, t, pm_counter.measures->energy.watts[i]);
	t+=inc_time;
      }
    }
  } else {	//If all lines will be printed
    if (set > pm_counter.measures->energy.watts_sets_size-1 || set <-1)
      return -1;
	
    line_t p_lines;
    LINE_AND(&p_lines, lines, pm_counter.lines);
    n_lines_counter= 0;
    n_lines_print= 0;

    for (i=0; i<__NLINEBITS && n_lines_print < pm_counter.measures->energy.lines_len; i++) {
      if(LINE_ISSET( i, &p_lines ))          n_lines_print++;
      if(LINE_ISSET( i, &pm_counter.lines )) n_lines_counter++;
    }

    ind_print=(int *)malloc( n_lines_print*sizeof(int));
    ind_lines=(int *)malloc( n_lines_print*sizeof(int));

    j= 0; ii= 0;
    for (i=0; i<__NLINEBITS && j < pm_counter.measures->energy.lines_len; i++) {
      if(LINE_ISSET( i, &p_lines ) && LINE_ISSET( i, &pm_counter.lines )) {
        ind_print[ii]= j;
	ind_lines[ii]= i;
	ii++;
	j++;
      } else if(!LINE_ISSET( i, &p_lines ) && LINE_ISSET( i, &pm_counter.lines ))
	j++;
    }

    interval=pm_counter.measures->energy.watts_sets[pm_counter.measures->energy.watts_sets_size-1]-pm_counter.measures->energy.watts_sets[0];

    if (set == -1) {
      init= 0;
      last= pm_counter.measures->energy.watts_sets_size-1;
    } else {
      init= set;
      last= set+1;
    }

    int offset = 0;
    watts_size = pm_counter.measures->energy.watts_size;

    for( s= init; s < last; s++ ) {
      ini=pm_counter.measures->energy.watts_sets[s];
      fin=pm_counter.measures->energy.watts_sets[s+1];

      time=pm_counter.measures->timing[(s*2)+1]-pm_counter.measures->timing[s*2];
      inc_time=time/(fin-ini-1);

      interval = fin-ini;

      t=0.0;
      for(i=0; i<interval; i++) {
        sum = 0.0;

	for(j=0;j<n_lines_print;j++)
	  sum+=pm_counter.measures->energy.watts[offset + ( i+interval*ind_print[j])];

	for (m=0; m < nmeasures; m++)
	  measures[m] += pm_counter.measures->energy.watts[offset + ( i+interval*ind_print[m])]; 

	measures[nmeasures] += sum;

	n_reads++;
	t+=inc_time;
      }

      offset+= (n_lines_counter*interval);

    }

    free(ind_print);
    free(ind_lines);
  
  }

  for (m=0; m <= nmeasures; m++) 
    measures[m] = measures[m] / (double)n_reads;

  return(0);
}
