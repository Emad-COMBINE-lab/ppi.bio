---
title: How-To Guide for Proteome-Wide PPI Inference
---

Using the online ppi.bio interface, users are capable of making inferences between:

1. a user-defined protein and 
2. all the proteins of a user-specified organism.

![Proteome-Wide Diagram]({STATIC_URL}imgs/preteome_wide_diagram.svg)

### Fill out the "Proteome-Wide Prediction" form

To begin, visit the "[Proteome-wide Prediction](/infer/proteome/submit)" page, where you'll be presented with a form 
asking you for:

1. The **organism** whose proteome to use. Interactions between your submitted amino acid sequence and the proteins of this organism will be computed.
2. The **inference model** to use to infer <abbr title="Protein-Protein Interactions">PPI</abbr>s. Currently, [RAPPPID](https://doi.org/10.1093/bioinformatics/btac429) and INTREPPPID are supported.
3. The **amino acid sequence** of a protein whose interactions you're interesting on inferring. <abbr title="Protein-Protein Interactions">PPI</abbr>s between the protein encoded by the user-inputted amino acid sequence and the proteome proteins will be inferred.
4. Hit "Predict Interactions" to start inferring.
 
![Proteome-Wide Submission Form Screenshot]({STATIC_URL}imgs/proteome_wide_form.webp)

Your task will be added to a queue. Once accepted, a progress bar will appear which shows how your job is going.


