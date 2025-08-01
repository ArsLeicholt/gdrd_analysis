library(ggmsa)
library(ggplot2)

protein_sequences <- "Gdrd_ClustalO_Alignment_visualize.fasta"

p <- ggmsa(protein_sequences, char_width = 0.5, seq_name = TRUE, color = "Chemistry_AA") + 
  geom_seqlogo(color = "Chemistry_AA") +
  theme(axis.text.y = element_text(face = "bold", color = "black"),
        axis.text.x = element_text(face = "bold", color = "black"))

# Save as PNG with 300 DPI
ggsave("gdrd_msa_visualization.png", 
       plot = p, 
       width = 12, 
       height = 8, 
       dpi = 300, 
       units = "in")

print("MSA visualization saved as gdrd_msa_visualization.png")