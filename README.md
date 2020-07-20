# KalmanFilter
 Code that simulates particle showers and calculates their parameters.

## REVIEW

##### GI filter for a pads detector
##### TRAGALDABAS Version X.x.x

*****************************
>April 2020. ***JA Garzon***. labCAF / USC
>
>April 2020. *Sara Costa*.  
>July 2020. *Miguel Cruces*
*****************************


### GENE
It generates ntrack tracks from a charged particle and propagates them in 
the Z axis direction through NPLAN planes.
### DIGIT
It simulates the digital answer in NPLAN planes of detectors, in which
- the coordinates (nx,ny) of the crossed pad are determined
- the flight time is determined integrating tint
### GI
It reconstructs the track through the GI Filter
### Comments
Some coding criteria:
- The variable names, follow, in general, some mnemonic rule
- Names of vectors start with v
- Names of matrixes start with m
********************************************************************
> **Typical units:**  
> Mass, momentum and energy: *MeV*  
> Distances in *mm*  
> Time of *ps*
********************************************************************