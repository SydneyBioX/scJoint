# Package and functions

library(ggplot2)
library(ggthemes)
library(scattermore)
library(ggpubr)
library(Rtsne)
library(uwot)
library(pals)
library(grDevices)
library(optparse)



theme_plot <- function () {
    theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
        theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 12), 
              panel.border = element_rect(colour = "black", fill = NA, 
                                          size = 1.2))
}


pal <- function (n, pal = NULL, remove_black = FALSE) {
    
    if (n <= 36) {
        col <- pals::polychrome(n)
    }
    else {
        col <- pals::polychrome(36)
        col <- grDevices::colorRampPalette(col)(n)
    }
    names(col) <- NULL
    return(col)
}


# Arguments

option_list <- list(
    make_option(c("--output_dir"), type = "character", default = "output/", 
                help = "output folder name"),
    make_option(c("--input_dir"), type = "character", default = "input/", 
                help = "input folder name"),
    make_option(c("--TSNE"), type = "logical", default = TRUE, 
                help = "run TSNE"),
    make_option(c("--UMAP"), type = "logical", default = TRUE, 
                help = "run UMAP"),
    make_option(c("--proportion"), type = "numeric", default = 1, 
                help = "proportion of cells to include in visualisation")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)


# Reading label reference
label_class <- read.delim(file.path(opt$input_dir, "label_to_idx.txt"),
                          header = FALSE)
label_class_num <- unlist(lapply(strsplit(label_class$V1, " "),
                                 function(x) x[length(x)]))
label_class_name <- unlist(lapply(strsplit(label_class$V1, " "),
                                  function(x) paste(x[-length(x)], collapse = " ")))
label_class <- data.frame(name = label_class_name,
                          num = label_class_num)


# Proportion of cells to be visualised
proportion <- opt$proportion

# The folder with output
results_dir <- opt$output_dir

embedding_files <- list.files(results_dir, "embeddings.txt")

embedding <- list()
for (i in 1:length(embedding_files)) {
    embedding[[i]] <- read.delim(file.path(results_dir, embedding_files[i]),
                                 header = FALSE, sep = " ")
}

names(embedding) <- gsub("_embeddings.txt", "", embedding_files)

cat("Dimension of embedding: ")
print(lapply(embedding, dim))





# Reading KNN prediction

knn_prediction_files <- list.files(results_dir, pattern = "knn_predictions.txt")

knn_prediction <- list()
for (i in 1:length(knn_prediction_files)) {
    knn_prediction[[i]] <- read.delim(file.path(results_dir, knn_prediction_files[i]),
                                      header = FALSE, sep = " ")
    knn_prediction[[i]] <- label_class$name[knn_prediction[[i]]$V1 + 1]
}

names(knn_prediction) <- gsub("_knn_predictions.txt", "", knn_prediction_files)


rna_dataset <- setdiff(names(embedding), names(knn_prediction))
print(rna_dataset)
rna_prediction <- list()
for (i in 1:length(rna_dataset)) {
    rna_prediction[[i]] <- read.delim(file.path(results_dir, paste0(rna_dataset[i], "_predictions.txt")),
                                      header = FALSE, sep = " ")
    rna_prediction[[i]] <- label_class$name[apply(rna_prediction[[i]], 1, which.max)]
}

names(rna_prediction) <- rna_dataset


prediction_list <- append(rna_prediction, knn_prediction)
prediction_list <- prediction_list[names(embedding)]

batch <- rep(names(prediction_list), unlist(lapply(prediction_list, length)))
combine_embedding <- do.call(rbind, embedding)
prediction <- do.call(c, prediction_list)


idx <- sort(sample(length(batch), round(length(batch) * proportion)))
combine_embedding <- combine_embedding[idx, ]
prediction <- prediction[idx]
batch <- batch[idx]

cat("Dimension to be visualised: ")
print(dim(combine_embedding))

set.seed(2020)
cellType_color <- pal(length(unique(prediction)))
names(cellType_color) <- sort(unique(prediction))

set.seed(2020)
rand_idx <- sample(nrow(combine_embedding), nrow(combine_embedding))


if (opt$TSNE) {
    set.seed(2020)
    print("Running TSNE")
    tsne_res <- Rtsne::Rtsne(combine_embedding, pca = FALSE, verbose = TRUE, max_iter = 2000)

    df <- data.frame(tSNE1 = tsne_res$Y[, 1], tSNE2 = tsne_res$Y[, 2], 
                     prediction = prediction,
                     batch = batch)
    write.table(df, file = file.path(results_dir, "tsne_embedding.txt"), row.names = FALSE)
    
    
    
    g1 <- ggplot(df[rand_idx,], aes(x = tSNE1, y = tSNE2, color = batch)) +
        geom_scattermore(pointsize = 0.5) +
        scale_color_brewer(palette = "Set1") +
        theme_plot() +
        theme(aspect.ratio = 1, legend.position = "bottom") +
        labs(title = "Batch")
    
    
    g2 <- ggplot(df, aes(x = tSNE1, y = tSNE2, color = prediction)) +
        geom_scattermore(pointsize = 0.5) +
        scale_color_manual(values = cellType_color) +
        theme_plot() +
        theme(aspect.ratio = 1, legend.position = "bottom") +
        labs(title = "Predicted")
    
    
    
    ggarrange(g1, g2, ncol = 2, nrow = 1, align = "hv")
    ggsave(file.path(results_dir, "TSNE_plot.pdf"), width = 20, height = 20)
    
    
    
}

if (opt$UMAP) {
    set.seed(2020)
    print("Running UMAP")
    
    umap_res <- uwot::umap(combine_embedding, min_dist = 0.3)

    df <- data.frame(UMAP1 = umap_res[, 1], UMAP2 = umap_res[, 2],
                     prediction = prediction,
                     batch = batch)
    write.table(df, file = file.path(results_dir, "umap_embedding.txt"), row.names = FALSE)
    
    
    g1 <- ggplot(df[rand_idx, ], aes(x = UMAP1, y = UMAP2, color = batch)) +
        geom_scattermore(pointsize = 0.5) +
        scale_color_brewer(palette = "Set1") +
        theme_plot() +
        theme(aspect.ratio = 1, legend.position = "bottom") +
        labs(title = "Batch")
    
    
    g2 <- ggplot(df, aes(x = UMAP1, y = UMAP2, color = prediction)) +
        geom_scattermore(pointsize = 0.5) +
        scale_color_manual(values = cellType_color) +
        theme_plot() +
        theme(aspect.ratio = 1, legend.position = "bottom") +
        labs(title = "Predicted")
    
    ggarrange(g1, g2, ncol = 2, nrow = 1, align = "hv")
    ggsave(file.path(results_dir, "UMAP_plot.pdf"), width = 20, height = 20)
    
    
    
}

