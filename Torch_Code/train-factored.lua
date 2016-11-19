require 'nn'
require 'nngraph'
require 'hdf5'

dofile 'data.lua'
dofile 'models.lua'
dofile 'model_utils.lua'
dofile 'DCCA.lua'

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/demo-train.hdf5', [[Path to the training *.hdf5 file from preprocess.py]])
cmd:option('-val_data_file','data/demo-val.hdf5', [[Path to validation *.hdf5 file from preprocess.py]])
cmd:option('-data_file_2','data/demo-train.hdf5', [[Path to the training *.hdf5 file from preprocess.py]])
cmd:option('-val_data_file_2','data/demo-val.hdf5', [[Path to validation *.hdf5 file from preprocess.py]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as
                                             savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is
                                             the validation perplexity]])
cmd:option('-num_shards', 0, [[If the training data has been broken up into different shards,
                             then training files are in this many partitions]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-load_key_vecs', 0, [[if == 1, load keywords]])

--CCA specs
cmd:option('-rcov1', 0, [[rcov1 for CCA]])
cmd:option('-rcov2', 0, [[rcov2 for CCA]])
cmd:option('-cca_k', 100, [[K for CCA]])
cmd:option('-cca', 0, [[use CCA]])
cmd:option('-cca_start', 7, [[start using cca from this epoch]])

-- rnn model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-factored', 0, [[if factored == 1,train factored lstm decoder]])
cmd:option('-factor_size', 500, [[size of factored state]])
cmd:option('-fine_tune_start', 8, [[if test1 == 1, will not train the second model in supervised way ]])
cmd:option('-test1', 0, [[if test1 == 1, will not train the second model in supervised way ]])
cmd:option('-test3', 0, [[if test1 == 1, will not train the decoder 2 ]])
cmd:option('-test4', 0, [[if test1 == 1, will not train the encoder 2 ]])
cmd:option('-test1', 0, [[if test1 == 1, will not train the second model in supervised way ]])
cmd:option('-test3', 0, [[if test3 == 1, will not train the second model in supervised way ]])
cmd:option('-test4', 0, [[if test4 == 1, will not train the second model in supervised way ]])
cmd:option('-joint', 0, [[joint equals 1 means train model jointly ]])
cmd:option('-mix_ratio', 0.5, [[ratio of joint and orginal model ]])
cmd:option('-std_iter', 10, [[iteration for standerd task]])

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-attn', 1, [[If = 1, use attention on the decoder side. If = 0, it uses the last
                       hidden state of the decoder as context at each time step.]])
cmd:option('-brnn', 0, [[If = 1, use a bidirectional RNN. Hidden states of the fwd/bwd RNNs are summed.]])
cmd:option('-use_chars_enc', 0, [[If = 1, use character on the encoder side (instead of word embeddings]])
cmd:option('-use_chars_dec', 0, [[If = 1, use character on the decoder side (instead of word embeddings]])
cmd:option('-reverse_src', 0, [[If = 1, reverse the source sequence. The original
                              sequence-to-sequence paper found that this was crucial to
                              achieving good performance, but with attention models this
                              does not seem necessary. Recommend leaving it to 0]])
cmd:option('-init_dec', 1, [[Initialize the hidden/cell state of the decoder at time
                           0 to be the last hidden/cell state of the encoder. If 0,
                           the initial states of the decoder are set to zero vectors]])
cmd:option('-input_feed', 1, [[If = 1, feed the context vector at each time step as additional
                             input (vica concatenation with the word embeddings) to the decoder]])
cmd:option('-multi_attn', 0, [[If > 0, then use a another attention layer on this layer of
                             the decoder. For example, if num_layers = 3 and `multi_attn = 2`,
                             then the model will do an attention over the source sequence
                             on the second layer (and use that as input to the third layer) and
                             the penultimate layer]])
cmd:option('-res_net', 0, [[Use residual connections between LSTM stacks whereby the input to
                          the l-th LSTM layer if the hidden state of the l-1-th LSTM layer
                          added with the l-2th LSTM layer. We didn't find this to help in our
                          experiments]])

cmd:text("")
cmd:text("Below options only apply if using the character model.")
cmd:text("")

-- char-cnn model specs (if use_chars == 1)
cmd:option('-char_vec_size', 25, [[Size of the character embeddings]])
cmd:option('-kernel_width', 6, [[Size (i.e. width) of the convolutional filter]])
cmd:option('-num_kernels', 1000, [[Number of convolutional filters (feature maps). So the
                                 representation from characters will have this many dimensions]])
cmd:option('-num_highway_layers', 2, [[Number of highway layers in the character model]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

-- optimization
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd (vanilla SGD), adagrad, adadelta, adam]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings: sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1]])
cmd:option('-learning_rate_2', 0.5, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings: sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                             on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-lr_decay_2', 0.1, [[Decay learning rate by this much if (i) perplexity does not decrease
                      on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-curriculum', 1, [[For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])
cmd:option('-max_batch_l', '', [[If blank, then it will infer the max batch size from validation
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

cmd:option('-start_symbol', 0, [[Use special start-of-sentence and end-of-sentence tokens
                               on the source side. We've found this to make minimal difference]])
-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-gpuid2', -1, [[If this is >= 0, then the model will use two GPUs whereby the encoder
                          is on the first GPU and the decoder is on the second GPU.
                          This will allow you to train with bigger batches/models.]])
cmd:option('-cudnn', 0, [[Whether to use cudnn or not for convolutions (for the character model).
                        cudnn has much faster convolutions so this is highly recommended
                        if using the character model]])
-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-valid_every', 1000, [[validate the model after this much epoch]])

function zero_table(t)
  for i = 1, #t do
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      if i == 1 or (opt.joint == 1 and i == 4) then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuid2)
      end
    end
    t[i]:zero()
  end
end

function train(train_data, valid_data)

  local timer = torch.Timer()
  local num_params = 0
  local start_decay = 0
  params, grad_params = {}, {}
  opt.train_perf = {}
  opt.val_perf = {}
  opt.train_perf_2 = {}
  opt.val_perf_2 = {}

  for i = 1, #layers do
    if opt.gpuid2 >= 0 then
      if i == 1 or (opt.joint == 1 and i == 4) then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuid2)
      end
    end
    local p, gp = layers[i]:getParameters()
    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
  end

  if opt.pre_word_vecs_enc:len() > 0 then
    local f = hdf5.open(opt.pre_word_vecs_enc)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vec_layers[1].weight[i]:copy(pre_word_vecs[i])
    end
  end
  if opt.pre_word_vecs_dec:len() > 0 then
    local f = hdf5.open(opt.pre_word_vecs_dec)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vec_layers[2].weight[i]:copy(pre_word_vecs[i])
    end
  end
  if opt.brnn == 1 then --subtract shared params for brnn
    num_params = num_params - word_vec_layers[1].weight:nElement()
    word_vec_layers[3].weight:copy(word_vec_layers[1].weight)
    if opt.use_chars_enc == 1 then
      for i = 1, charcnn_offset do
        num_params = num_params - charcnn_layers[i]:nElement()
        charcnn_layers[i+charcnn_offset]:copy(charcnn_layers[i])
      end
    end
  end
  print("Number of parameters: " .. num_params)

  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
    word_vec_layers[1].weight[1]:zero()
    if opt.joint == 1 then
       word_vec_layers[3].weight[1]:zero()
     end
     cutorch.setDevice(opt.gpuid2)
     word_vec_layers[2].weight[1]:zero()
     if opt.joint == 1 then
       word_vec_layers[4].weight[1]:zero()
     end
  else
    word_vec_layers[1].weight[1]:zero()
    word_vec_layers[2].weight[1]:zero()
    if opt.brnn == 1 then
      word_vec_layers[3].weight[1]:zero()
    end
  end

  -- prototypes for gradients so there is no need to clone
  encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  encoder_bwd_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  context_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  if opt.cca == 1 then
    context_proto_cca = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  end
  -- need more copies of the above if using two gpus
  if opt.gpuid2 >= 0 then
    encoder_grad_proto2 = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
    context_proto2 = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
    encoder_bwd_grad_proto2 = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  end
  decoder_clones = {}
  encoder_clones = {}
  generator_box = {}
  criterion_box = {}
  table.insert(generator_box, generator)
  table.insert(criterion_box,criterion)
  if opt.joint == 1 then
    table.insert(generator_box,generator_2)
    table.insert(criterion_box,criterion_2)
  end
  -- clone encoder/decoder up to max source/target length
  encoder_clones[1] = clone_many_times(encoder, opt.max_sent_l_src)
  if opt.joint == 1 then
    decoder_clones[1] = clone_many_times(decoder, opt.max_sent_l)
  else
    decoder_clones[1] = clone_many_times(decoder, opt.max_sent_l_targ)
  end
  if opt.brnn == 1 then
     encoder_bwd_clones = clone_many_times(encoder_bwd, opt.max_sent_l_src)
  end
  if opt.joint == 1 then
    decoder_clones[2] = clone_many_times(decoder, opt.max_sent_l)
    encoder_clones[2] = clone_many_times(encoder_2, opt.max_sent_l)
  end
  for i = 1, opt.max_sent_l_src do
    if encoder_clones[1][i].apply then
      encoder_clones[1][i]:apply(function(m) m:setReuse() end)
    end
    if opt.brnn == 1 then
      encoder_bwd_clones[i]:apply(function(m) m:setReuse() end)
    end
  end
  if opt.joint == 1 then
    for i = 1, opt.max_sent_l_src_2 do
      if encoder_clones[2][i].apply then
        encoder_clones[2][i]:apply(function(m) m:setReuse() end)
      end
    end
  end
  if opt.joint == 1 then
    for i = 1, opt.max_sent_l do
      for j = 1, 2 do
        if  decoder_clones[j][i].apply then
         decoder_clones[j][i]:apply(function(m) m:setReuse() end)
        end
      end
    end
  else
    for i = 1, opt.max_sent_l_targ do
      if decoder_clones[1][i].apply then
       decoder_clones[1][i]:apply(function(m) m:setReuse() end)
      end
    end
  end

  local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init = h_init:cuda()
    cutorch.setDevice(opt.gpuid)
    if opt.cca == 1 then
      context_proto_cca = context_proto_cca:cuda()
    end
    if opt.gpuid2 >= 0 then
      encoder_grad_proto2 = encoder_grad_proto2:cuda()
      encoder_bwd_grad_proto2 = encoder_bwd_grad_proto2:cuda()
      context_proto = context_proto:cuda()
      cutorch.setDevice(opt.gpuid2)
      encoder_grad_proto = encoder_grad_proto:cuda()
      encoder_bwd_grad_proto = encoder_bwd_grad_proto:cuda()
      context_proto2 = context_proto2:cuda()
      cutorch.setDevice(opt.gpuid)
    else
      context_proto = context_proto:cuda()
      encoder_grad_proto = encoder_grad_proto:cuda()
      if opt.brnn == 1 then
        encoder_bwd_grad_proto = encoder_bwd_grad_proto:cuda()
      end
    end
  end

  -- these are initial states of encoder/decoder for fwd/bwd steps
  init_fwd_enc = {}
  init_bwd_enc = {}
  init_fwd_dec = {}
  init_bwd_dec = {}
  init_fwd_enc_cca = {}
  init_bwd_enc_cca = {}

  for L = 1, opt.num_layers do
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
    if opt.cca == 1 then
      table.insert(init_fwd_enc_cca, h_init:clone())
      table.insert(init_fwd_enc_cca, h_init:clone())
      table.insert(init_bwd_enc_cca, h_init:clone())
      table.insert(init_bwd_enc_cca, h_init:clone())
    end
  end
  if opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid2)
  end
  if opt.input_feed == 1 then
    table.insert(init_fwd_dec, h_init:clone())
  end
  table.insert(init_bwd_dec, h_init:clone())
  for L = 1, opt.num_layers do
    table.insert(init_fwd_dec, h_init:clone())
    table.insert(init_fwd_dec, h_init:clone())
    table.insert(init_bwd_dec, h_init:clone())
    table.insert(init_bwd_dec, h_init:clone())
  end
  dec_offset = 3 -- offset depends on input feeding
  if opt.input_feed == 1 then
    dec_offset = dec_offset + 1
  end

  --(self) return state of this function is copied
  function reset_state(state, batch_l, t)
    if t == nil then
      local u = {}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u, state[i][{{1, batch_l}}])
      end
      return u
    else
      local u = {[t] = {}}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u[t], state[i][{{1, batch_l}}])
      end
      return u
    end
  end

  -- clean layer before saving to make the model smaller
  function clean_layer(layer)
    if opt.gpuid >= 0 then
      layer.output = torch.CudaTensor()
      layer.gradInput = torch.CudaTensor()
    else
      layer.output = torch.DoubleTensor()
      layer.gradInput = torch.DoubleTensor()
    end
    if layer.modules then
      for i, mod in ipairs(layer.modules) do
        clean_layer(mod)
      end
    elseif torch.type(self) == "nn.gModule" then
      layer:apply(clean_layer)
    end
  end

  -- decay learning rate if val perf does not improve or we hit the opt.start_decay_at limit
  function decay_lr(epoch)
    print(opt.val_perf)
    if epoch >= opt.start_decay_at then
      start_decay = 1
    end

    if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf-1] ~= nil then
      local curr_ppl = opt.val_perf[#opt.val_perf]
      local prev_ppl = opt.val_perf[#opt.val_perf-1]
      if curr_ppl > prev_ppl then
        start_decay = 1
      end
    end
    if start_decay == 1 then
      opt.learning_rate = opt.learning_rate * opt.lr_decay
    end
  end

  function train_batch(d, i, encoder_idx, decoder_idx, epoch, train_loss, train_nonzeros, num_words_target, num_words_source, start_time)
    zero_table(grad_params, 'zero')
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]
    local keyword
    if opt.factored == 1 then
      keyword = d[9]:transpose(1,2)
    end

    local encoder_grads = encoder_grad_proto[{{1, batch_l}, {1, source_l}}]
    local encoder_bwd_grads
    if opt.brnn == 1 then
      encoder_bwd_grads = encoder_bwd_grad_proto[{{1, batch_l}, {1, source_l}}]
    end
    if opt.gpuid >= 0 then
      cutorch.setDevice(opt.gpuid)
    end
    local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
    local context = context_proto[{{1, batch_l}, {1, source_l}}]
    -- forward prop encoder
    if encoder_idx ~= decoder_idx then
      dec_iter = source_l
    else
      dec_iter = target_l
    end
    for t = 1, source_l do
      encoder_clones[encoder_idx][t]:training()
      local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
      local out = encoder_clones[encoder_idx][t]:forward(encoder_input)
      rnn_state_enc[t] = out
      context[{{},t}]:copy(out[#out])
    end

    local rnn_state_enc_bwd
    if opt.brnn == 1 then
      rnn_state_enc_bwd = reset_state(init_fwd_enc, batch_l, source_l+1)
      for t = source_l, 1, -1 do
        encoder_bwd_clones[t]:training()
        local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
        local out = encoder_bwd_clones[t]:forward(encoder_input)
        rnn_state_enc_bwd[t] = out
        context[{{},t}]:add(out[#out])
      end
    end

    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2)
      local context2 = context_proto2[{{1, batch_l}, {1, source_l}}]
      context2:copy(context)
      context = context2
    end
    -- copy encoder last hidden state to decoder initial state
    local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 0)
    if opt.init_dec == 1 then
      for L = 1, opt.num_layers do
        rnn_state_dec[0][L*2-1+opt.input_feed]:copy(rnn_state_enc[source_l][L*2-1])
        rnn_state_dec[0][L*2+opt.input_feed]:copy(rnn_state_enc[source_l][L*2])
      end
      if opt.brnn == 1 then
        for L = 1, opt.num_layers do
          rnn_state_dec[0][L*2-1+opt.input_feed]:add(rnn_state_enc_bwd[1][L*2-1])
          rnn_state_dec[0][L*2+opt.input_feed]:add(rnn_state_enc_bwd[1][L*2])
        end
      end
    end
    -- forward prop decoder
    local preds = {}
    local source_copy
    local decoder_input
    for t = 1, dec_iter do
      decoder_clones[decoder_idx][t]:training()
      local decoder_input
      if encoder_idx ~= decoder_idx then
        source_copy = source:clone()
        if opt.attn == 1 then
          decoder_input = {source_copy[t], context, table.unpack(rnn_state_dec[t-1])}
        else
          decoder_input = {source_copy[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
        end
      else
        if opt.attn == 1 then
          decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
        else
          decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
        end
        if opt.factored == 1 then
          table.insert(decoder_input, keyword)
        end
      end
      local out = decoder_clones[decoder_idx][t]:forward(decoder_input)
      local next_state = {}
      table.insert(preds, out[#out])
      if opt.input_feed == 1 then
        table.insert(next_state, out[#out])
      end
      for j = 1, #out-1 do
        table.insert(next_state, out[j])
      end
      rnn_state_dec[t] = next_state
    end

    -- backward prop decoder
    encoder_grads:zero()
    if opt.brnn == 1 then
      encoder_bwd_grads:zero()
    end

    local drnn_state_dec = reset_state(init_bwd_dec, batch_l)
    local loss = 0
    for t = dec_iter, 1, -1 do
      local pred = generator_box[decoder_idx]:forward(preds[t])
      local dl_dpred
      if encoder_idx ~= decoder_idx then
        loss = loss + criterion_box[decoder_idx]:forward(pred, source_copy[t])/batch_l
        dl_dpred = criterion_box[decoder_idx]:backward(pred, source_copy[t])
      else
        -- print (pred:size(), target_out:size(), decoder_idx)
        loss = loss + criterion_box[decoder_idx]:forward(pred, target_out[t])/batch_l
        dl_dpred = criterion_box[decoder_idx]:backward(pred, target_out[t])
      end
      dl_dpred:div(batch_l)
      local dl_dtarget = generator_box[decoder_idx]:backward(preds[t], dl_dpred)
      drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)
      local decoder_input
      if encoder_idx ~= decoder_idx then
        if opt.attn == 1 then
          decoder_input = {source_copy[t], context, table.unpack(rnn_state_dec[t-1])}
        else
          decoder_input = {source_copy[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
        end
      else
        if opt.attn == 1 then
          decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
        else
          decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
        end
        if opt.factored == 1 then
          table.insert(decoder_input, keyword)
        end
      end
      local dlst = decoder_clones[decoder_idx][t]:backward(decoder_input, drnn_state_dec)
      -- accumulate encoder/decoder grads
      if opt.attn == 1 then
        encoder_grads:add(dlst[2])
        if opt.brnn == 1 then
          encoder_bwd_grads:add(dlst[2])
        end
      else
        encoder_grads[{{}, source_l}]:add(dlst[2])
        if opt.brnn == 1 then
          encoder_bwd_grads[{{}, 1}]:add(dlst[2])
        end
      end
      drnn_state_dec[#drnn_state_dec]:zero()
      if opt.input_feed == 1 then
        drnn_state_dec[#drnn_state_dec]:add(dlst[3])
      end
      if opt.factored == 1 then
        for j = dec_offset, #dlst-1 do
          drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
        end
      else
        for j = dec_offset, #dlst do
          drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
        end
      end
    end

    word_vec_layers[2 + (decoder_idx - 1)*2].gradWeight[1]:zero()
    if opt.fix_word_vecs_dec == 1 then
      word_vec_layers[2 + (decoder_idx - 1)*2].gradWeight:zero()
    end

    local grad_norm = 0
    grad_norm = grad_norm + grad_params[2 + (decoder_idx - 1)*3]:norm()^2 + grad_params[3 + (decoder_idx - 1)*3]:norm()^2

    -- backward prop encoder
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
      local encoder_grads2 = encoder_grad_proto2[{{1, batch_l}, {1, source_l}}]
      encoder_grads2:zero()
      encoder_grads2:copy(encoder_grads)
      encoder_grads = encoder_grads2 -- batch_l x source_l x rnn_size
    end

    local drnn_state_enc = reset_state(init_bwd_enc, batch_l)
    if opt.init_dec == 1 then
      for L = 1, opt.num_layers do
        drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
        drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
      end
    end

    for t = source_l, 1, -1 do
      local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
      if opt.attn == 1 then
        drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
      else
        if t == source_l then
          drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
        end
      end
      local dlst = encoder_clones[encoder_idx][t]:backward(encoder_input, drnn_state_enc)
      for j = 1, #drnn_state_enc do
        drnn_state_enc[j]:copy(dlst[j+1])
      end
    end

    if opt.brnn == 1 then
      local drnn_state_enc = reset_state(init_bwd_enc, batch_l)
      if opt.init_dec == 1 then
        for L = 1, opt.num_layers do
          drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
          drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
        end
      end
      for t = 1, source_l do
        local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
        if opt.attn == 1 then
          drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grads[{{},t}])
        else
          if t == 1 then
            drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grads[{{},t}])
          end
        end
        local dlst = encoder_bwd_clones[t]:backward(encoder_input, drnn_state_enc)
        for j = 1, #drnn_state_enc do
          drnn_state_enc[j]:copy(dlst[j+1])
        end
      end
    end

    word_vec_layers[1 + (encoder_idx - 1)*2].gradWeight[1]:zero()
    if opt.fix_word_vecs_enc == 1 then
      word_vec_layers[1 + (encoder_idx - 1)*2].gradWeight:zero()
    end

    grad_norm = grad_norm + grad_params[1 + (encoder_idx - 1)*3]:norm()^2
    if opt.brnn == 1 then
      grad_norm = grad_norm + grad_params[4]:norm()^2
    end
    grad_norm = grad_norm^0.5
    if opt.brnn == 1 then
      word_vec_layers[1].gradWeight:add(word_vec_layers[3].gradWeight)
      if opt.use_chars_enc == 1 then
        for j = 1, charcnn_offset do
          charcnn_grad_layers[j]:add(charcnn_grad_layers[j+charcnn_offset])
        end
      end
    end
    -- Shrink norm and update params
    local param_norm = 0
    local shrinkage = opt.max_grad_norm / grad_norm
    local update_idx = #grad_params
    if opt.joint == 1 then
      update_idx = 3
    end
    for j = 1, update_idx do
      if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
        if j == 1 then
          cutorch.setDevice(opt.gpuid)
        else
          cutorch.setDevice(opt.gpuid2)
        end
      end
      if shrinkage < 1 then
        if opt.joint == 1 then
          if j == 1 then
            grad_params[j + (encoder_idx - 1)*3]:mul(shrinkage)
          else
            grad_params[j + (decoder_idx - 1)*3]:mul(shrinkage)
          end
        else
          grad_params[j]:mul(shrinkage)
        end
      end
      if opt.optim == 'adagrad' then
        adagrad_step(params[j], grad_params[j], layer_etas[j], optStates[j])
      elseif opt.optim == 'adadelta' then
        adadelta_step(params[j], grad_params[j], layer_etas[j], optStates[j])
      elseif opt.optim == 'adam' then
        adam_step(params[j], grad_params[j], layer_etas[j], optStates[j])
      else
        if opt.joint == 1 then
          if j == 1 then
            params[j + (encoder_idx - 1)*3]:add(grad_params[j + (encoder_idx - 1)*3]:mul(-opt.learning_rate))
          else
            params[j + (decoder_idx - 1)*3]:add(grad_params[j + (decoder_idx - 1)*3]:mul(-opt.learning_rate))
          end
        else
          params[j]:add(grad_params[j]:mul(-opt.learning_rate))
        end
      end
      if opt.joint == 1 then
        if j == 1 then
          param_norm = param_norm + params[j + (encoder_idx - 1)*3]:norm()^2
        else
          param_norm = param_norm + params[j + (decoder_idx - 1)*3]:norm()^2
        end
      else
        param_norm = param_norm + params[j]:norm()^2
      end
    end
    param_norm = param_norm^0.5
    if opt.brnn == 1 then
      word_vec_layers[3].weight:copy(word_vec_layers[1].weight)
      if opt.use_chars_enc == 1 then
        for j = 1, charcnn_offset do
          charcnn_layers[j+charcnn_offset]:copy(charcnn_layers[j])
        end
      end
    end

    -- Bookkeeping
    num_words_target = num_words_target + batch_l*target_l
    num_words_source = num_words_source + batch_l*source_l
    train_nonzeros = train_nonzeros + nonzeros
    train_loss = train_loss + loss*batch_l
    local time_taken = timer:time().real - start_time
    if i % opt.print_every == 0 then
      local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
        epoch, i, train_data:size(), batch_l, opt.learning_rate)
      stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
        math.exp(train_loss/train_nonzeros), param_norm, grad_norm)
      stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
        (num_words_target+num_words_source) / time_taken,
        num_words_source / time_taken,
        num_words_target / time_taken)
      print(stats)
    end
    if i % 50 == 0 then
      collectgarbage()
    end
    return train_loss, train_nonzeros, num_words_target, num_words_source
  end

  function train_cca(d, epoch, i)
    local start_time = timer:time().real
    zero_table(grad_params, 'zero')
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]
    if opt.gpuid >= 0 then
      cutorch.setDevice(opt.gpuid)
    end
    local target_cca = target:clone()
    local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
    local context = context_proto[{{1, batch_l}, {1, source_l}}]
    -- forward prop encoder
    for t = 1, source_l do
      encoder_clones[1][t]:training()
      local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
      local out = encoder_clones[1][t]:forward(encoder_input)
      rnn_state_enc[t] = out
      context[{{},t}]:copy(out[#out])
    end
    local rnn_state_enc_cca = reset_state(init_fwd_enc_cca, batch_l, 0)
    local context_cca = context_proto_cca[{{1, batch_l}, {1, target_l}}]
    for t = 1, target_l do
      encoder_clones[2][t]:training()
      local encoder_input = {target_cca[t],table.unpack(rnn_state_enc_cca[t-1])}
      local out = encoder_clones[2][t]:forward(encoder_input)
      rnn_state_enc_cca[t] = out
      context_cca[{{},t}]:copy(out[#out])
    end
    local input1 = torch.CudaTensor(context:size()[1], context:size()[3]):copy(context[{{},source_l}])
    local input2 = torch.CudaTensor(context_cca:size()[1], context_cca:size()[3]):copy(context_cca[{{},target_l}])
    local corrl = CCA_model:forward({input1, input2})
    local dDCCA = CCA_model:backward({input1, input2})
    --copy the last state of encoder for back-prop
    local drnn_state_enc = reset_state(init_bwd_enc, batch_l)
    local drnn_state_enc_cca = reset_state(init_bwd_enc_cca, batch_l)
    --for L = 1, opt.num_layers do
    --  drnn_state_enc[L*2-1]:copy(rnn_state_enc[source_l][L*2-1])
    --  drnn_state_enc[L*2]:copy(rnn_state_enc[source_l][L*2])
    --  drnn_state_enc_cca[L*2-1]:copy(rnn_state_enc_cca[source_l][L*2-1])
    --  drnn_state_enc_cca[L*2]:copy(rnn_state_enc_cca[source_l][L*2])
    --end
    --back-prop encoder1
    for t = source_l, 1, -1 do
      local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
      if t == source_l then
        drnn_state_enc[#drnn_state_enc]:add(dDCCA[1])
      end
      local dlst = encoder_clones[1][t]:backward(encoder_input, drnn_state_enc)
      for j = 1, #drnn_state_enc do
        drnn_state_enc[j]:copy(dlst[j+1])
      end
    end
    --back-prop encoder2
    for t = target_l, 1, -1 do
      local encoder_input = {target_cca[t], table.unpack(rnn_state_enc_cca[t-1])}
      if t == source_l then
        drnn_state_enc_cca[#drnn_state_enc_cca]:add(dDCCA[2])
      end
      local dlst = encoder_clones[2][t]:backward(encoder_input, drnn_state_enc_cca)
      for j = 1, #drnn_state_enc do
        drnn_state_enc_cca[j]:copy(dlst[j+1])
      end
    end

    word_vec_layers[1].gradWeight[1]:zero()
    word_vec_layers[3].gradWeight[1]:zero()
    local grad_norm = 0
    grad_norm = grad_norm + grad_params[1]:norm()^2 + grad_params[4]:norm()^2
    grad_norm = grad_norm^0.5
    -- Shrink norm and update params
    local param_norm = 0
    local shrinkage = opt.max_grad_norm / grad_norm*2
    if shrinkage < 1 then
      grad_params[1]:mul(shrinkage)
      grad_params[4]:mul(shrinkage)
    end
    if opt.optim == 'adagrad' then
      adagrad_step(params[1], grad_params[1], layer_etas[1], optStates[1])
      adagrad_step(params[4], grad_params[4], layer_etas[4], optStates[4])
    elseif opt.optim == 'adadelta' then
      adadelta_step(params[1], grad_params[1], layer_etas[1], optStates[1])
      adadelta_step(params[4], grad_params[4], layer_etas[4], optStates[4])
    elseif opt.optim == 'adam' then
      adam_step(params[1], grad_params[1], layer_etas[1], optStates[1])
      adam_step(params[4], grad_params[4], layer_etas[4], optStates[4])
    else
      params[1]:add(grad_params[1]:mul(-opt.learning_rate))
      params[4]:add(grad_params[4]:mul(-opt.learning_rate))
    end
    if i%opt.print_every == 0 then
      local time_taken = timer:time().real - start_time
      local num_words_target = batch_l*target_l
      local num_words_source = batch_l*source_l
      local stats = string.format('CCA model spec: Training: %d/%d/%d total/source/target tokens/sec',
        (num_words_target+num_words_source) / time_taken,
        num_words_source / time_taken,
        num_words_target / time_taken)
      print(stats)
    end
  end

  function train_epoch(data, epoch)
    local train_nonzeros = 0
    local train_loss = 0
    local batch_order = torch.randperm(data.length) -- shuffle mini batch order
    local start_time = timer:time().real
    local num_words_target = 0
    local num_words_source = 0
    local encoder_idx, decoder_idx = 1, 1

    for i = 1, data:size() do
      train_batch(data, i, encoder_idx, decoder_idx,epoch)
    end
    return train_loss, train_nonzeros
  end

  local total_loss, total_nonzeros, batch_loss, batch_nonzeros
  for epoch = opt.start_epoch, opt.epochs do
    generator:training()
    if opt.joint == 1 then
      generator_2:training()
    end
    if opt.num_shards > 0 then
      total_loss = 0
      total_nonzeros = 0
      local shard_order = torch.randperm(opt.num_shards)
      for s = 1, opt.num_shards do
        local fn = train_data .. '.' .. shard_order[s] .. '.hdf5'
        print('loading shard #' .. shard_order[s])
        local shard_data = data.new(opt, fn)
        batch_loss, batch_nonzeros = train_epoch(shard_data, epoch)
        total_loss = total_loss + batch_loss
        total_nonzeros = total_nonzeros + batch_nonzeros
      end
    else
      -- total_loss, total_nonzeros = train_epoch(train_data, epoch)
      local train_nonzeros = 0
      local train_loss = 0
      local train_nonzeros_2 = 0
      local train_loss_2 = 0
      local train_nonzeros_3 = 0
      local train_loss_3 = 0
      local train_nonzeros_4 = 0
      local train_loss_4 = 0
      local batch_order = torch.randperm(train_data.length)
      local batch_order_2
      if opt.joint == 1 then
        batch_order_2 = torch.randperm(train_data_2.length)
      end
      local start_time = timer:time().real
      local num_words_target = 0
      local num_words_source = 0
      local num_words_target_2 = 0
      local num_words_source_2 = 0
      local num_words_target_3 = 0
      local num_words_source_3 = 0
      local num_words_target_4 = 0
      local num_words_source_4 = 0
      local data_size = train_data:size()
      if opt.joint == 1 then
        data_size = math.min(train_data:size(), train_data_2:size())
      end
      for i = 1, data_size do
        local d
        if epoch <= opt.curriculum then
          d = train_data[i]
        else
          d = train_data[batch_order[i]]
        end
        encoder_idx = 1
        decoder_idx = 1
        if i%opt.print_every == 0 and opt.joint == 1 then
          print ('first model specs')
        end
        train_loss, train_nonzeros, num_words_target, num_words_source = train_batch(d, i, encoder_idx, decoder_idx, epoch, train_loss, train_nonzeros, num_words_target, num_words_source, start_time)
        if i%opt.valid_every == 0 then
          local valid_loss = eval(valid_data, 1)
          local stats = string.format('Validation: Epoch: %d, Batch: %d/%d',   epoch, i, train_data:size())
          stats = stats .. string.format('Loss: %f', valid_loss)
          print(stats)
        end
        if opt.joint == 1 then
          if opt.test1 == 0 then
            encoder_idx = 2
            decoder_idx = 2
            local d_2
            if epoch <= opt.curriculum then
              d_2 = train_data_2[i]
            else
              d_2 = train_data_2[batch_order_2[i]]
            end
            if i%opt.print_every == 0 then
              print ('second model specs')
            end
            train_loss_2, train_nonzeros_2, num_words_target_2, num_words_source_2 = train_batch(d_2, i, encoder_idx, decoder_idx, epoch, train_loss_2, train_nonzeros_2, num_words_target_2, num_words_source_2, start_time)
          end
          -- add CCA here for the time being
          if opt.cca == 1 then
            if (i-opt.std_iter/2)%opt.std_iter == 0 and (epoch > opt.cca_start - 1) then

              local cca_idx = 0
              local batch_order_cca = torch.randperm(train_data.length)
              for j = 1,opt.std_iter*opt.mix_ratio do
                local d_cca = train_data[batch_order_cca[j]]
                train_cca(d_cca,epoch, i)
              end
            end
          end
          -- when doing joint train, train a en-en
          if i%opt.std_iter == 0 and opt.fine_tune_start >= epoch then
            start_time_2 = timer:time().real
            local en_en_idx = 0
            local de_de_idx = 0
            local batch_order_3 = torch.randperm(train_data.length)
            local batch_order_4 = torch.randperm(train_data_2.length)
            for j = 1,opt.std_iter*opt.mix_ratio do
              encoder_idx = 1
              decoder_idx = 2
            if opt.test3 == 0 then
              local d_3 = train_data[batch_order_3[j]]
              if j%opt.print_every == 0 then
                print ('joint 1 (speed might be off)')
              end
              train_loss_3, train_nonzeros_3, num_words_target_3, num_words_source_3 = train_batch(d_3, j, encoder_idx, decoder_idx, epoch, train_loss_3, train_nonzeros_3, num_words_target_3, num_words_source_3, start_time_2)
            end
            if opt.test4 == 0 then
              encoder_idx = 2
              decoder_idx = 1
              local d_4 = train_data_2[batch_order_4[j]]
              if j%opt.print_every == 0 then
                print ('joint 2 (speed might be off)')
              end
              train_loss_4, train_nonzeros_4, num_words_target_4, num_words_source_4 = train_batch(d_4, j, encoder_idx, decoder_idx, epoch, train_loss_4, train_nonzeros_4, num_words_target_4, num_words_source_4, start_time_2)
            end
            end
          end
        end
      end

      total_loss = train_loss
      total_nonzeros = train_nonzeros
      if opt.joint == 1 and opt.test1 == 0 then
        total_loss_2 = train_loss_2
        total_nonzeros_2 = train_nonzeros_2
      end
    end
    local train_score = math.exp(total_loss/total_nonzeros)
    local train_score_2
    print('Train', train_score)
    if opt.joint == 1 and opt.test1 == 0 then
      train_score_2 = math.exp(total_loss_2/total_nonzeros_2)
      print('Joint Train', train_score_2)
      opt.train_perf_2[#opt.train_perf_2 + 1] = train_score_2
    end
    opt.train_perf[#opt.train_perf + 1] = train_score
    local score = eval(valid_data, 1)
    local score_2
    if opt.joint == 1 and opt.test1 == 0 then
      score_2 = eval(valid_data_2, 2)
    end
    opt.val_perf[#opt.val_perf + 1] = score
    if opt.optim == 'sgd' then --only decay with SGD
      decay_lr(epoch)
    end
    -- clean and save models
    local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, score)
    if epoch % opt.save_every == 0 then
      print('saving checkpoint to ' .. savefile)
      clean_layer(generator)
      if opt.joint == 1 then
        clean_layer(generator_2)
        torch.save(savefile, {{encoder, decoder, generator, encoder_2, decoder_2, generator_2}, opt})
      else
        if opt.brnn == 0 then
          torch.save(savefile, {{encoder, decoder, generator}, opt})
        else
          torch.save(savefile, {{encoder, decoder, generator, encoder_bwd}, opt})
        end
      end
    end
  end
  -- save final model
  local savefile = string.format('%s_final.t7', opt.savefile)
  clean_layer(generator)
  print('saving final model to ' .. savefile)
  if opt.joint == 1 then
    clean_layer(generator_2)
    torch.save(savefile, {{encoder:double(), decoder:double(), generator:double(),
        encoder_2:double(), decoder_2:double(), generator_2:double(),}, opt})
  else
    if opt.brnn == 0 then
      torch.save(savefile, {{encoder:double(), decoder:double(), generator:double()}, opt})
    else
      torch.save(savefile, {{encoder:double(), decoder:double(), generator:double(),
            encoder_bwd:double()}, opt})
    end
  end
end

function eval(data, model_idx)
  encoder_clones[model_idx][1]:evaluate()
  decoder_clones[model_idx][1]:evaluate() -- just need one clone
  generator_box[model_idx]:evaluate()
  if opt.brnn == 1 then
    encoder_bwd_clones[1]:evaluate()
  end

  local nll = 0
  local total = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]
    local keyword
    if opt.factored == 1 then
      keyword = d[9]:transpose(1,2)
    end
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
    end
    local rnn_state_enc = reset_state(init_fwd_enc, batch_l)
    local context = context_proto[{{1, batch_l}, {1, source_l}}]
    -- forward prop encoder
    for t = 1, source_l do
      local encoder_input = {source[t], table.unpack(rnn_state_enc)}
      local out = encoder_clones[model_idx][1]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:copy(out[#out])
    end

    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2)
      local context2 = context_proto2[{{1, batch_l}, {1, source_l}}]
      context2:copy(context)
      context = context2
    end

    local rnn_state_dec = reset_state(init_fwd_dec, batch_l)
    if opt.init_dec == 1 then
      for L = 1, opt.num_layers do
        rnn_state_dec[L*2-1+opt.input_feed]:copy(rnn_state_enc[L*2-1])
        rnn_state_dec[L*2+opt.input_feed]:copy(rnn_state_enc[L*2])
      end
    end

    if opt.brnn == 1 then
      local rnn_state_enc = reset_state(init_fwd_enc, batch_l)
      for t = source_l, 1, -1 do
        local encoder_input = {source[t], table.unpack(rnn_state_enc)}
        local out = encoder_bwd_clones[1]:forward(encoder_input)
        rnn_state_enc = out
        context[{{},t}]:add(out[#out])
      end
      if opt.init_dec == 1 then
        for L = 1, opt.num_layers do
          rnn_state_dec[L*2-1+opt.input_feed]:add(rnn_state_enc[L*2-1])
          rnn_state_dec[L*2+opt.input_feed]:add(rnn_state_enc[L*2])
        end
      end
    end

    local loss = 0
    for t = 1, target_l do
      local decoder_input
      if opt.attn == 1 then
        decoder_input = {target[t], context, table.unpack(rnn_state_dec)}
      else
        decoder_input = {target[t], context[{{},source_l}], table.unpack(rnn_state_dec)}
      end
      if opt.factored == 1 then
        table.insert(decoder_input, keyword)
      end
      local out = decoder_clones[model_idx][1]:forward(decoder_input)
      rnn_state_dec = {}
      if opt.input_feed == 1 then
        table.insert(rnn_state_dec, out[#out])
      end
      for j = 1, #out-1 do
        table.insert(rnn_state_dec, out[j])
      end
      local pred = generator_box[model_idx]:forward(out[#out])
      loss = loss + criterion_box[decoder_idx]:forward(pred, target_out[t])
    end
    nll = nll + loss
    total = total + nonzeros
  end
  local valid = math.exp(nll / total)
  print("Valid", valid)
  collectgarbage()
  return valid
end

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs_dec' then
      table.insert(word_vec_layers, layer)
    elseif layer.name == 'word_vecs_enc' then
      table.insert(word_vec_layers, layer)
    elseif layer.name == 'charcnn_enc' or layer.name == 'mlp_enc' then
      local p, gp = layer:parameters()
      for i = 1, #p do
        table.insert(charcnn_layers, p[i])
        table.insert(charcnn_grad_layers, gp[i])
      end
    end
  end
end

function main()
  -- parse input params
  opt = cmd:parse(arg)

  torch.manualSeed(opt.seed)

  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    if opt.gpuid2 >= 0 then
      print('using CUDA on second GPU ' .. opt.gpuid2 .. '...')
    end
    require 'cutorch'
    require 'cunn'
    if opt.cudnn == 1 then
      print('loading cudnn...')
      require 'cudnn'
    end
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
  end

  -- Create the data loader class.
  print('loading data...')
  if opt.num_shards == 0 then
    train_data = data.new(opt, opt.data_file)
  else
    train_data = opt.data_file
  end
  if opt.joint == 1 then
    print('loading data for Joint model...')
    if opt.num_shards == 0 then
      train_data_2 = data.new(opt, opt.data_file_2)
    else
      train_data_2 = opt.data_file_2
    end
    valid_data_2 = data.new(opt, opt.val_data_file_2)
  end

  valid_data = data.new(opt, opt.val_data_file)
  print('done!')
  print(string.format('Source vocab size: %d, Target vocab size: %d',
      valid_data.source_size, valid_data.target_size))
  if opt.joint == 1 then
    print(string.format('Joint Source vocab size: %d, Target vocab size: %d',
         valid_data_2.source_size, valid_data_2.target_size))
  end
  opt.max_sent_l_src = valid_data.source:size(2)
  opt.max_sent_l_targ = valid_data.target:size(2)
  if opt.joint == 1 then
    opt.max_sent_l_src_2 = valid_data_2.source:size(2)
    opt.max_sent_l_targ_2 = valid_data_2.target:size(2)
  end
  opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_targ)
  if opt.joint == 1 then
    opt.max_sent_l = math.max(opt.max_sent_l,opt.max_sent_l_src_2,opt.max_sent_l_targ_2)
  end
  if opt.max_batch_l == '' then
    opt.max_batch_l = valid_data.batch_l:max()
  end

  if opt.use_chars_enc == 1 or opt.use_chars_dec == 1 then
    opt.max_word_l = valid_data.char_length
  end
  print(string.format('Source max sent len: %d, Target max sent len: %d',
      valid_data.source:size(2), valid_data.target:size(2)))
  if opt.joint == 1 then
    print(string.format('Joint Source max sent len: %d, Target max sent len: %d',
         valid_data_2.source:size(2), valid_data_2.target:size(2)))
  end
  -- Build model
  if opt.train_from:len() == 0 then
    encoder = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
    if opt.factored == 1 then
      decoder = make_factored_lstm(valid_data, opt, 'dec', opt.use_chars_enc)
    else
      decoder = make_lstm(valid_data, opt, 'dec', opt.use_chars_dec)
    end
    generator, criterion = make_generator(valid_data, opt)
    if opt.brnn == 1 then
      encoder_bwd = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
    end
    if opt.joint == 1 then
      encoder_2 = make_lstm(valid_data_2, opt, 'enc', opt.use_chars_enc)
      decoder_2 = make_lstm(valid_data_2, opt, 'dec', opt.use_chars_enc)
      generator_2, criterion_2 = make_generator(valid_data_2, opt)
    end
  else
    assert(path.exists(opt.train_from), 'checkpoint path invalid')
    print('loading ' .. opt.train_from .. '...')
    local checkpoint = torch.load(opt.train_from)
    local model, model_opt = checkpoint[1], checkpoint[2]
    opt.num_layers = model_opt.num_layers
    opt.rnn_size = model_opt.rnn_size
    opt.input_feed = model_opt.input_feed
    opt.attn = model_opt.attn
    opt.brnn = model_opt.brnn
    encoder = model[1]:double()
    decoder = model[2]:double()
    generator = model[3]:double()
    if model_opt.brnn == 1 then
      encoder_bwd = model[4]:double()
    end
    if model_opt.joint == 1 then
      encoder_2 = model[4]:double()
      decoder_2 = model[5]:double()
      generator_2 = model[6]:double()
    end
    _, criterion = make_generator(valid_data, opt)
  end

  if opt.cca == 1 then
    CCA_model = nn.DCCA(opt)
  end

  layers = {encoder, decoder, generator}
  if opt.brnn == 1 then
    table.insert(layers, encoder_bwd)
  end
  if opt.joint == 1 then
    table.insert(layers,encoder_2)
    table.insert(layers,decoder_2)
    table.insert(layers,generator_2)
  end

  if opt.optim ~= 'sgd' then
    layer_etas = {}
    optStates = {}
    for i = 1, #layers do
      layer_etas[i] = opt.learning_rate -- can have layer-specific lr, if desired
      optStates[i] = {}
    end
  end

  if opt.gpuid >= 0 then
    for i = 1, #layers do
      if opt.gpuid2 >= 0 then
        if i == 1 or i == 4 then
          cutorch.setDevice(opt.gpuid) --encoder on gpu1
        else
          cutorch.setDevice(opt.gpuid2) --decoder/generator on gpu2
        end
      end
      layers[i]:cuda()
    end
    if opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2) --criterion on gpu2
    end
    criterion:cuda()
    if opt.joint == 1 then criterion_2:cuda()
      criterion_2:cuda()
    end
  end

  -- these layers will be manipulated during training
  word_vec_layers = {}
  if opt.use_chars_enc == 1 then
    charcnn_layers = {}
    charcnn_grad_layers = {}
  end
  encoder:apply(get_layer)
  decoder:apply(get_layer)
  if opt.brnn == 1 then
    if opt.use_chars_enc == 1 then
      charcnn_offset = #charcnn_layers
    end
    encoder_bwd:apply(get_layer)
  end
  if opt.joint == 1 then
    encoder_2:apply(get_layer)
    decoder_2:apply(get_layer)
  end
  if opt.joint == 1 then
    train(train_data,valid_data,train_data_2,valid_data_2)
  else
    train(train_data, valid_data)
  end
end

main()
