# KalmanFilter
 Code that simulates particle showers and reconstructs their tracks from 
 their fingerprints on the TRAGALDABAS detector.
 
 This detector is composed by four plates/plans with 120 cells each one, 
 capable of detect particles. By linking those fingerprints, we want to 
 reconstruct the initial track using Kalman Filter.

## REVIEW

##### GI filter for a pads detector
##### TRAGALDABAS Version X.x.x

*****************************
>April 2020. ***JA Garzon***. labCAF / USC
>
>April 2020. *Sara Costa*.  
>July 2020. *Miguel Cruces*
*****************************


### GenerateTracks class
It generates ntrack tracks from a charged particle and propagates them in 
the Z axis direction through NPLAN planes.
#### GenerateTracks.digitization() method
It simulates the digital answer in NPLAN planes of detectors, in which:
- the coordinates (nx,ny) of the crossed pad are determined.
- the flight time is determined integrating tint.
### KalmanFilter class
It reconstructs the track through the GI Filter-
### Comments
Some coding criteria:
- The variable names, follow, in general, some mnemonic rules
- Names of vectors start with **v**
- Names of matrixes start with **m**
- Names of indices start with **i**
- Names of numbers start with **n**
********************************************************************
> **Typical units:**  
> Mass, momentum and energy in *MeV*  
> Distances in *mm*  
> Time in *ps*
********************************************************************