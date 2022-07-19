import torch

t1 = torch.tensor([[1, 5, 10], [4, 65, 6], [13, 5, 3]])
print('TENSOR ONE \n', t1)

#Get all max values and extract them
mx, idxs = torch.max(t1, dim = 1)
idxs.unsqueeze_(dim = 1)
mx.unsqueeze_(dim = 1)
tensor_mask = t1.scatter_(1, idxs, 0)
end_tensor = t1[tensor_mask.bool()].view(3, 2)
print(end_tensor)

permuted_cols = torch.randperm(end_tensor.shape[0])
permuted_logits = end_tensor[permuted_cols]

#Use torch.cat to split each row tensor at the index of the max position and concantenate
argmax_added_tensor = torch.cat((permuted_logits, mx), dim = 1)
print(argmax_added_tensor)
#Iterate over each row and add the max value at the given index
output = torch.zeros(0, 3)
for (row_idx, row), (sample_number, max_val), (max_index_pos, max_index) in zip(enumerate(argmax_added_tensor), enumerate(mx), enumerate(idxs)):
    print("ROW: \n", row)
    first_section = torch.cat((row[:max_index.item()].clone().detach(), max_val), dim = 0)
    print('FIRST SECTION: \n', first_section)
    print(row.shape[0] - 2)
    second_section = row[max_index.item():row.shape[0]-1].clone().detach()
    print('SECOND SECTION: \n', second_section)
    new_row = torch.cat((first_section, second_section), dim = 0)
    print('NEW ROW: \n', new_row)
    new_row.unsqueeze_(dim = 0)
    print(new_row.shape)
    print(output.shape)
    output = torch.cat((output, new_row), dim = 0)

print('FINAL OUTPUT: \n', output)

# print('PERMUTED: \n', torch.where(t1 != mx, torch.where(permuted_cols != idxs, t1[permuted_cols]) t1))