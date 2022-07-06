import torch
from torch.autograd import Function

#CWTM (Confidence Weighted by Teacher Max) Distillation Loss - subclass of torch.autograd.Function, all methods are static
class CWTM_DistillationLoss(Function):

    #Define forward method (where we compute the loss) - take in the student predictions, teacher predictions, and true predictions for each model
    @staticmethod
    def forward(ctx, s_preds, t_preds, true_preds):
        #Save both prediction tensors into context object for gradient computations
        ctx.save_for_backward(s_preds, t_preds, true_preds)
        #Use Negative Log Liklihood loss - we are passing in the softmax-corrected activations as opposed to the raw logits
        loss = torch.nn.NLLLoss()
        return loss(s_preds, t_preds)

    #Define backward method (where the gradient of the loss is computed)
    @staticmethod
    def backward(ctx, grad_output):
        #Implement Equation 10 from the BAN paper https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf
        #Obtain labels from the saved tensors
        s_smax_preds, t_smax_preds, true_preds = ctx.saved_tensors
        #Find the predicted labels from both the student and the teacher for that sample
        s_pred_labels = torch.max(s_smax_preds, dim = 1)
        t_pred_labels = torch.max(t_smax_preds, dim = 1)
        #Find the difference between the STUDENT predicted labels and the GROUND TRUTH predicted labels
        diff = torch.subtract(s_pred_labels, true_preds)
        #Find the SUM of all the teacher labels
        t_label_sum = torch.cumsum(t_pred_labels)
        #Divide each element in s_pred_labels by the total teacher sum
        weight_tensor = torch.divide(t_pred_labels, t_label_sum)
        #Multiply the weight tensor by the gradients to get the final gradient update, normalize by batch size (first element in the tensor)
        batch_size = s_pred_labels.shape[0].item()
        grad_input = batch_size * torch.mul(weight_tensor, diff)
        return grad_input, grad_output