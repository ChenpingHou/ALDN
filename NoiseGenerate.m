function noise_label = NoiseGenerate(label,p)
Class = unique(label);
Label = label;
for i = 1:length(Class)
    index = find(Label==i);
    num_select = floor(length(index)*p);
    rand_index = randperm(length(index));
    index_select = index(rand_index(1:num_select));
    class = Class;
    class(class==i) = [];
    label(index_select) = randsrc(num_select,1,class');
end
noise_label = label;
            