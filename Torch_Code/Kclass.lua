require 'nn'
require 'nngraph'
require 'hdf5'

dofile 'data.lua'
dofile 'models.lua'
dofile 'model_utils.lua'
cmd = torch.CmdLine()

cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/demo-train.hdf5', [[Path to the training *.hdf5 file from preprocess.py]])
cmd:option('-val_data_file','data/demo-val.hdf5', [[Path to validation *.hdf5 file from preprocess.py]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as
                                             savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is
                                             the validation perplexity]])
cmd:option('-fix_encoder', 0, [[if fix_encoder is 1, then use pretrained encoder]])

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

cmd:option('-load_key_vecs', 0, [[if == 1, load keywords]])
cmd:option('-valid_every', 500, [[validate model after this much minibatch]])

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

function calculate_loss(input, target)
  return torch.sum(torch.cmul(target, input:log())+torch.cmul(1-target, (1-input):log()))
end

function train(train_data, valid_data)

  local timer = torch.Timer()
  local num_params = 0
  local start_decay = 0
  params, grad_params = {}, {}
  opt.train_perf = {}
  opt.val_perf = {}

  for i = 1, #layers do
  	local p, gp = layers[i]:getParameters()
    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
  end
  print("Number of parameters: " .. num_params)

  word_vec_layers[1].weight[1]:zero()
  encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  encoder_clones = clone_many_times(encoder, opt.max_sent_l_src)
  for i = 1, opt.max_sent_l_src do
    if encoder_clones[i].apply then
      encoder_clones[i]:apply(function(m) m:setReuse() end)
    end
  end

  local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init = h_init:cuda()
    cutorch.setDevice(opt.gpuid)
    encoder_grad_proto = encoder_grad_proto:cuda()
  end

  -- encoder states
  init_fwd_enc = {}
  init_bwd_enc = {}
   for L = 1, opt.num_layers do
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
  end

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


  for epoch = opt.start_epoch, opt.epochs do
    generator:training()
    local train_nonzeros = 0
    local train_loss = 0
    local batch_order = torch.randperm(train_data.length)
    for i = 1, train_data:size() do
    	-- take batch out
    	local d
      if epoch <= opt.curriculum then
        d = train_data[i]
      else
        d = train_data[batch_order[i]]
      end
      zero_table(grad_params, 'zero')
      local source, nonzeros = d[4], d[3]
      local batch_l, source_l = d[5], d[7]
      local keyword = d[9]:transpose(1,2)
      local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
      local context
      -- forward prop encoder
	    for t = 1, source_l do
	      encoder_clones[t]:training()
	      local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
	      local out = encoder_clones[t]:forward(encoder_input)
	      rnn_state_enc[t] = out
	      if t == source_l then
	      	context = out[#out]
	      end
	    end
			-- predict
	    local pred = generator:forward(context)
            if opt.gpuid < 0 then
      pred = torch.DoubleTensor():resize(pred:size()):copy(pred)
      keyword = torch.DoubleTensor():resize(keyword:size()):copy(keyword)
            end
	    local loss = criterion:forward(pred, keyword)
	    train_loss = train_loss + loss
	    local dl_dpred = criterion:backward(pred, keyword)
      local dl_dtarget = generator:backward(context, dl_dpred)
      --print(dl_dtarget)
      
	 		
	    local drnn_state_enc = reset_state(init_bwd_enc, batch_l)
      drnn_state_enc[#drnn_state_enc]:add(dl_dtarget)

	    --back-prop encoder
	    for t = source_l, 1, -1 do
	      local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
	      local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
	      for j = 1, #drnn_state_enc do
	        drnn_state_enc[j]:copy(dlst[j+1])
	      end
	    end
            if opt.fix_encoder == 1 then
              word_vec_layers[1].gradWeight:zero()
              grad_params[1]:zero()
            else 
	      word_vec_layers[1].gradWeight[1]:zero()
            end
            
	    local grad_norm = grad_params[1]:norm()^2 + grad_params[2]:norm()^2
            grad_norm = grad_norm^0.5
	    local param_norm = 0
    	local shrinkage = opt.max_grad_norm / grad_norm*2
        local update_idx = #grad_params
        for j = 1, update_idx do
    	if shrinkage < 1 then
	      grad_params[j]:mul(shrinkage)
	    end
	    if opt.optim == 'adagrad' then
	      adagrad_step(params[j], grad_params[j], layer_etas[j], optStates[j])
	    elseif opt.optim == 'adadelta' then
	      adadelta_step(params[j], grad_params[j], layer_etas[j], optStates[j])
	    elseif opt.optim == 'adam' then
	      adam_step(params[j], grad_params[j], layer_etas[j], optStates[j])
	    else
	      params[j]:add(grad_params[j]:mul(-opt.learning_rate))
	    end
        end
	    if i%opt.print_every == 0 then
    		local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
        epoch, i, train_data:size(), batch_l, opt.learning_rate)
        stats = stats .. string.format('Loss: %f, |GParam|: %f', loss, grad_norm)
        print(stats)
    	end
        if i%opt.valid_every == 0 then
          local valid_loss = eval(valid_data)
          local stats = string.format('Validation: Epoch: %d, Batch: %d/%d',   epoch, i, train_data:size())
          stats = stats .. string.format('Loss: %f', valid_loss)
          print(stats)
        end
    end -- batch
    epoch_loss = train_loss/train_data:size()
    print('Train', epoch_loss)
    opt.train_perf[#opt.train_perf + 1] = epoch_loss
    local eval_loss = eval(valid_data)
    opt.val_perf[#opt.val_perf + 1] = score
    if opt.optim == 'sgd' then --only decay with SGD
      decay_lr(epoch)
    end
    local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, eval_loss)
    if epoch % opt.save_every == 0 then
      print('saving checkpoint to ' .. savefile)
      clean_layer(generator)
      torch.save(savefile, {{encoder, generator}, opt})
    end
  end -- epoch
  local savefile = string.format('%s_final.t7', opt.savefile)
  clean_layer(generator)
  print('saving final model to ' .. savefile)
  torch.save(savefile, {{encoder:double(), generator:double()}, opt})
end --trainning function

function eval(data)
  encoder_clones[1]:evaluate()
  generator:evaluate()
  local nll = 0
  local total_loss = 0
  local sent_num = 0
  for i = 1, data:size() do
    local d = data[i]
    local source = d[4]
    local batch_l, source_l, keyword = d[5], d[7], d[9]:transpose(1,2) 
    local context
    local rnn_state_enc = reset_state(init_fwd_enc, batch_l)
    for t = 1, source_l do
      local encoder_input = {source[t], table.unpack(rnn_state_enc)}
      local out = encoder_clones[1]:forward(encoder_input)
      rnn_state_enc = out
      if t == source_l then
        context = out[#out]
      end
    end
    local pred = generator:forward(context)
    if opt.gpuid < 0 then
      pred = torch.DoubleTensor():resize(pred:size()):copy(pred)
      keyword = torch.DoubleTensor():resize(keyword:size()):copy(keyword)
    end
    local loss = criterion:forward(pred, keyword)
    total_loss = total_loss + loss
    sent_num = sent_num + batch_l
  end

  return total_loss/sent_num
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

  opt.load_key_vecs = 1

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

  print('loading data...')

  -- Create the data loader class.
  train_data = data.new(opt, opt.data_file)
  valid_data = data.new(opt, opt.val_data_file)
  print('done!')
  print(string.format('Source vocab size: %d, Target vocab size: %d',
    valid_data.source_size, valid_data.target_size))
  opt.max_sent_l_src = valid_data.source:size(2)
  opt.max_sent_l_targ = valid_data.target:size(2)
  opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_targ)
  if opt.max_batch_l == '' then
    opt.max_batch_l = valid_data.batch_l:max()
  end
  print(string.format('Source max sent len: %d, Target max sent len: %d',
    valid_data.source:size(2), valid_data.target:size(2)))

  -- Build model
  if fix_encoder == 0 then
  if opt.train_from:len() == 0 then
    encoder = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
    generator, criterion = keyword_generator(valid_data[1][9], opt)
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
    generator = model[3]:double()
    _, criterion = make_generator(valid_data, opt)
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
    generator, criterion = keyword_generator(valid_data[1][9], opt)
  end

  layers = {encoder, generator}

  if opt.optim ~= 'sgd' then
    layer_etas = {}
    optStates = {}
    for i = 1, #layers do
      layer_etas[i] = opt.learning_rate -- can have layer-specific lr, if desired
      optStates[i] = {}
    end
  end

  -- load everything to gpu
  if opt.gpuid >= 0 then
    for i = 1, #layers do
    	layers[i]:cuda()
    end
    criterion:cuda()
  end
  word_vec_layers = {}
  encoder:apply(get_layer)
  train(train_data, valid_data)
end

main()
