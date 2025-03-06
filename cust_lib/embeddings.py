import tensorflow as tf
from sentence_transformers import SentenceTransformer, util
import seaborn as sns


def get_embeddings(in_chunks_list, in_sentence_transformer_model=""):
    """
    Create embeddings from text chunks. 

    Args:
        in_chunks_list: a list of text chunks to be converted into embedding vectors
        in_sentence_transformer_model: optional preferred embedding model. See
            https://www.sbert.net/docs/sentence_transformer/pretrained_models.html for options
    Returns:
        A list of embedding vectors
    """
    if (in_sentence_transformer_model == ""):
        model_name = "all-MiniLM-L6-v2"
        #model_name = "multi-qa-MiniLM-L6-dot-v1"
    else:
         model = in_sentence_transformer_model

    model = SentenceTransformer(model_name)
    embedding_list = []
    embedding_list = model.encode(in_chunks_list)

    return embedding_list

def findSimlarEmbedding(in_query_embedding, in_kb_embedding_list, verbose=False):
    """
    Create embeddings from text chunks. 

    Args:
        in_query_embedding: xyz
        in_kb_embedding_list: List of knowledge base embeddings

    Returns:
        The embedding that best mateches the query embedding
    """
    similarities = []

    # Loop through the headline embeddings.
    for i, curr_kb_embedding  in enumerate(in_kb_embedding_list):
        curr_cosign_sim_score = __getCosignSimlarity(in_query_embedding, curr_kb_embedding)
        similarities.append({"index": i,
                        "embedding": curr_kb_embedding,
                       "cosign_similarity_score":curr_cosign_sim_score})
    
    similarities.sort(key = lambda similarities: similarities['cosign_similarity_score'],reverse=True)
    return similarities[0]
    


def __getCosignSimlarity(in_vector_1, in_vector_2):
    """
    This is a PLACEHOLDER function. Calculates the cosign simlarity value for two vectors

    Args:
        in_vector_1
        in_vector_2

    Returns:
        The cosign similarity value
    """
    cosine_similarity_score = util.cos_sim(in_vector_1, in_vector_2)

    return cosine_similarity_score[0][0]


# def visualizeEmbeedingCosignSimilarityScores(in_df):
"""
        Calculates the cosign simlarity value for two vectors

        Args:
            in_df: Dataframe containing all cosign similariies
        Returns:
            The cosign similarity value
        """

  #      sns.heatmap(in_df, cmap="winter")