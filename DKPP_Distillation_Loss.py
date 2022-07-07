import torch
from torch.autograd import Function

#DKPP (Dark Knowledge with Permuted Predictions) Distillation Loss - purpose is to test if the Dark Knowledge terms in Distillation Loss
#Are truly critical via permuting the dark knowledge logits so as to destroy the covariance matrix between max and 
class DKPP_DistillationLoss(Function):

    #The only required arguments for DKPP loss are the student and teacher predictions - the ground truth is not used as no importance weighting is done
    #The purpose is to test the Dark Knowledge Hypothesis, meaning that the student should be able to learn entirely form the DK passed on from the teacher
    @staticmethod
    def forward(ctx, s_preds, t_preds):
        #Save both tensors into context for use in backward method
        ctx.save_for_backward(s_preds, t_preds)
        #Loss between both is, as with the CWTM Distillation, simply the negative log liklihood
        loss = torch.nn.NLLLoss()
        return loss(s_preds, t_preds)

    @staticmethod
    def backward(ctx, grad_output):
        #Implementing equation 11 from the BAN paper https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf
        #Compute the gradients as usual (and scale based on batch size), but subtract via gradients of permuted teacher predictions
        #First, draw tensors from the context object
        s_smax_preds, t_smax_preds = ctx.saved_tensors
        #Find the true predicted labels for both tensors (maximum of both probability distributions)
        s_pred_labels = torch.max(s_smax_preds, dim = 1)
        t_pred_labels = torch.max(t_smax_preds, dim = 1)
        #Subtract both tensors and normalize by 1/b (where b is batch size) to obtain first term of gradient expression
        batch_size = s_pred_labels.shape[0].item()
        first_grad_term = (1 / batch_size) * torch.subtract(s_pred_labels, t_pred_labels)
        #Next, subtract the DK student logits by the permuted DK teacher logits and normalize once again to obtain second gradient term
        #Access the indexes of the first dimension (the index of the rows) and use .randperm to shuffle indexes
        #Then, simply re-arrange the tensor with the new indexes to obtain the shuffled tensor
        permuted_indexes = torch.randperm(t_smax_preds.shape[0])
        permuted_DK_terms = t_smax_preds[permuted_indexes]
        second_grad_term = (1 / batch_size) * torch.subtract(s_smax_preds, permuted_DK_terms)
        #Add the first and second terms together and return as grad_input
        grad_input = torch.add(first_grad_term, second_grad_term)
        return grad_input, grad_output