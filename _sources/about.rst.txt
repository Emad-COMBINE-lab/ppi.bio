About
=====

PPI.bio is a web interface for INTREPPPID and RAPPPID, two deep-learning models for predicting
`protein-protein interactions <https://en.wikipedia.org/wiki/Protein%E2%80%93protein_interaction>`_ (PPIs). This online
interface allows you to use INTREPPPID and RAPPPID through your browser.

INTREPPPID
----------
INTREPPPID stands for INcorporating TRiplet Error for Predicting Protein-Protein
Interactions using Deep learning. It's a "quintuplet" neural-network for predicting PPIs using
amino acids. It was designed to perform well on predicting the interaction between
proteins that belong to species other than the ones it was trained on.

RAPPPID
-------
RAPPPID stands for Regularised Automatic Prediction of Protein-Protein
Interactions using Deep learning. It's a twin neural-network for predicting PPIs
using only amino acids. It's designed specifically to address some limitations of existing
PPI prediction methods with regard to generalisation and out-of-distribution predictions.
You can read more about the RAPPPID model in our paper
"`RAPPPID: towards generalizable protein interaction prediction with AWD-LSTM twin networks <https://doi.org/10.1093/bioinformatics/btac429)>`_".

## PPI.bio
PPI.bio is a server that hosts a web interface which enables someone to use INTREPPPID or RAPPPID
using their browser. I host a version of PPI.bio at `PPI.bio <https://ppi.bio>`_ that is free
to use. That's the easiest way to make PPI predictions using the INTREPPPID or RAPPPID model. If, for
whatever reason, you should desire to host your own instance of PPI.bio, this
documentation should contain the information required to do so.
