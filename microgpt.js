// python: zip()
function array_zip(arr1, arr2) {
	var result = [];
	var max_length = Math.max(arr1.length, arr2.length);
	for (var i = 0; i < max_length; i++)
		result.push([arr1[i] || undefined, arr2[i] || undefined]);
	return result;
}

// python: [a + b for a, b in zip(array1, array2)]
function array_zip_sum(arr1, arr2) {
	var result = [];
	if(!arr1 || !arr2) debugger;
	var max_length = Math.max(arr1.length, arr2.length);
	for (var i = 0; i < max_length; i++)
		result.push(arr1[i].add(arr2[i]));
	return result;
}

// python: range()
function array_range(len) {
	var result = new Array(len);
	for(var i = 0; i < len; i++)
		result[i] = i;
	return result;
}

// python: shuffle()
function array_shuffle(arr) {
	var j, tmp;
	for (var i = arr.length - 1; i > 0; i--) {
		j = Math.floor(Math.random() * (i + 1));
		tmp = arr[j];
		arr[j] = arr[i];
		arr[i] = tmp;
	}
};

// python: random.gauss()
function random_gauss(mean, stdDev) {
	var u = 0, v = 0;
	while(u === 0) u = Math.random();
	while(v === 0) v = Math.random();
	return mean + stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// python: random.choices()
function random_choices(population, weights, k) {
    if (!population || population.length === 0) return [];
    var n = population.length;
    if (typeof k !== "number" || k <= 0) k = 1;

    if (!weights) {
      var resU = [];
      for (var i = 0; i < k; i++) {
        resU.push(population[Math.floor(Math.random() * n)]);
      }
      return resU;
    } else {
      if (weights.length !== n) throw new Error("weights length mismatch");
      var cum = new Array(n);
      var total = 0;
      for (var i = 0; i < n; i++) {
        var w = weights[i];
        if (w < 0) w = 0;
        total += w;
        cum[i] = total;
      }
      if (total <= 0) {
        var resZero = [];
        for (var j = 0; j < k; j++) {
          resZero.push(population[Math.floor(Math.random() * n)]);
        }
        return resZero;
      }

      var res = [];
      for (var j = 0; j < k; j++) {
        var r = Math.random() * total;
        var lo = 0, hi = n - 1, idx = n - 1;
        while (lo <= hi) {
          var mid = (lo + hi) >> 1;
          if (r < cum[mid]) { idx = mid; hi = mid - 1; } else { lo = mid + 1; }
        }
        res.push(population[idx]);
      }
      return res;
    }
};

// python: matrix()
function matrix(rows, cols, defaultValue) {
	var m = [];
	for (var i = 0; i < rows; i++) {
		m[i] = [];
		for (var j = 0; j < cols; j++) {
			m[i][j] = defaultValue;
		}
	}
	return m;
}

// python: print()
function print(msg) {
	if(output) {
		var new_line = document.createElement('DIV');
		new_line.innerHTML = msg;
		output.appendChild(new_line);
	}
	else
		print(msg);
}

var docs = input_txt.split('\n')
			.filter(function(str){ return str != '' && !str.match(/^ +$/); })
			.map(function(str){ return str.replace(/^ +/, '').replace(/ +$/, ''); });
array_shuffle(docs);
print("num docs: "+docs.length);

// Let there be a Tokenizer to translate strings to discrete symbols and back
var uchars = [];
for(var str of docs)
for(var i = str.length-1; i>=0; i--)
	if(uchars.indexOf(str[i]) == -1)
		uchars.push(str[i]);
var uchars = uchars.sort(); // unique characters in the dataset become token ids 0..n-1
print("uchars", uchars);
var BOS = uchars.length; // token id for the special Beginning of Sequence (BOS) token
var vocab_size = uchars.length + 1 // total number of unique tokens, +1 is for BOS
print("vocab size: "+vocab_size)

function Value(data, children, local_grads) {

	// scalar value of this node calculated during forward pass
	this.data = data;

	// derivative of the loss w.r.t. this node, calculated in backward pass
	this.grad = 0;

	// children of this node in the computation graph
	this._children = children || [];

	// local derivative of this node w.r.t. its children
	this._local_grads = local_grads || [];
}

Value.prototype.add = function(other) {
	if(other instanceof Value == false)
		other = new Value(other);

	return new Value(this.data + other.data, [this, other], [1, 1]);
}

Value.prototype.mul = function(other) {
	if(other instanceof Value == false)
		other = new Value(other);
	return new Value(this.data * other.data, [this, other], [other.data, this.data]);
}

Value.prototype.pow = function(other) {
	return new Value(this.data**other, [this], [other * this.data**(other-1)]);
}

Value.prototype.log = function(){
    return new Value(Math.log(this.data), [this], [1/this.data]);
}

Value.prototype.exp = function() {
	return new Value(Math.exp(this.data), [this], [Math.exp(this.data)]);
}

Value.prototype.relu = function() {
	return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1.0 : 0.0]);
}

// Value.prototype.neg = function() { return this.mul(-1); };
// Value.prototype.radd = function(other) { return this.add(other); }
Value.prototype.sub = function(other) {
	if(other instanceof Value == false)
		other = new Value(other);
	return new Value(this.data - other.data, [this, other], [1, 1]);
}
// Value.prototype.rsub = function(other) { debugger; return other + (-this); }
// Value.prototype.rmul = function(other) { return this.mul(other); }
Value.prototype.truediv = function(other) {
	if(other instanceof Value == false)
		return this.mul(other**-1);
	else
		return this.mul(other.pow(-1));
}
// Value.prototype.rtruediv = function(other) { return other * this**-1; }

Value.prototype.backward = function() {
	var topo = [];
	var visited = new Set();
	var build_topo = function(v) {
		if (visited.has(v) == false) {
			visited.add(v);
			for(var child of v._children)
				build_topo(child);
			topo.push(v);
		}
	};
	build_topo(this)
	this.grad = 1;
	for (var i = topo.length-1; i>=0; i--) {
		var v = topo[i];
		for (var ii = 0; ii < v._children.length; ii++) {
			var child = v._children[ii];
			var local_grad = v._local_grads[ii];
			child.grad += local_grad * v.grad;
		}
	}
}

// Initialize the parameters, to store the knowledge of the model.
var n_embd = 16;     // embedding dimension
var n_head = 4;      // number of attention heads
var n_layer = 1;     // number of layers
var block_size = 16; // maximum sequence length
var head_dim = Math.floor(n_embd / n_head); // dimension of each head
var matrix = function(nout, nin, std) {
    if (std === undefined) std = 0.08;
    var result = [];
    for (var i = 0; i < nout; i++) {
        var row = [];
        for (var j = 0; j < nin; j++) {
            row.push(new Value(random_gauss(0, std)));
        }
        result.push(row);
    }
    return result;
};
var state_dict = {
	'wte': matrix(vocab_size, n_embd),
	'wpe': matrix(block_size, n_embd),
	'lm_head': matrix(vocab_size, n_embd)
};

for(var i = 0; i < n_layer; i++) {
    state_dict['layer'+i+'.attn_wq'] = matrix(n_embd, n_embd)
    state_dict['layer'+i+'.attn_wk'] = matrix(n_embd, n_embd)
    state_dict['layer'+i+'.attn_wv'] = matrix(n_embd, n_embd)
    state_dict['layer'+i+'.attn_wo'] = matrix(n_embd, n_embd)
    state_dict['layer'+i+'.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict['layer'+i+'.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
}
// flatten params into a single list[Value]
var params = [];
for (var matrix_name in state_dict) {
	for (var p of state_dict[matrix_name].flat())
		params.push(p);
}
print("num params: "+params.length);

// Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next.
// Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
function linear(x, w) {
	var result = [];
	for (var wo of w) {
		var sum = new Value(0);
		for(var i = 0; i < x.length; i++) {
			var wi = wo[i];
			var xi = x[i];
			sum = sum.add(wi.mul(xi));
		}
		result.push(sum);
	}
	return result;
}

function softmax(logits) {

	var max_val = 0;
	for (var val of logits)
		max_val = Math.max(val.data, max_val);

	var exps = [];
	for (var val of logits) {
		if(!val.sub) debugger;
		exps.push(val.sub(max_val).exp());
	}

	var total = new Value(0);
	for(var val of exps)
		total = total.add(val);

	var result = [];
	for (var e of exps)
		result.push(e.truediv(total));

	return result;
}

function rmsnorm(x) {
	var sum = new Value(0);
	for (var xi of x)
		sum = sum.add(xi.mul(xi));
    var ms = sum.truediv(x.length);
    var scale = (ms.add(1e-5)).pow(-0.5);

	var result = [];
	for (var xi of x)
		result.push(xi.mul(scale));

    return result;
}

function gpt(token_id, pos_id, keys, values) {
	var tok_emb = state_dict['wte'][token_id]; // token embedding
    var pos_emb = state_dict['wpe'][pos_id]; // position embedding
    // joint token and position embedding
    var x = array_zip_sum(tok_emb, pos_emb);
    x = rmsnorm(x)

    for (var li = 0; li < n_layer; li++) {

		// 1) Multi-head attention block
        var x_residual = x;
        x = rmsnorm(x);
        var q = linear(x, state_dict['layer'+li+'.attn_wq']);
        var k = linear(x, state_dict['layer'+li+'.attn_wk']);
        var v = linear(x, state_dict['layer'+li+'.attn_wv']);
        keys[li].push(k);
        values[li].push(v);
        var x_attn = [];
        for (var h = 0; h < n_head; h++) {
            var hs = h * head_dim;
            var q_h = q.slice(hs, hs+head_dim);
			var k_h = [];
			for (var ki of keys[li]) k_h.push(ki.slice(hs, hs+head_dim));
			var v_h = [];
			for (var vi of values[li]) v_h.push(vi.slice(hs, hs+head_dim));
			var attn_logits = [];
			for (var t = 0; t < k_h.length; t++) {
				var sum = new Value(0);
				for (var j = 0; j < head_dim; j++) {
					sum = sum.add(q_h[j].mul(k_h[t][j]));
				}
				attn_logits.push(sum.truediv(head_dim**0.5));
			}
            var attn_weights = softmax(attn_logits);
			var head_out = [];
			for (var j = 0; j < head_dim; j++) {
				var sum = new Value(0);
				for (var t = 0; t < v_h.length; t++)
					sum = sum.add(attn_weights[t].mul(v_h[t][j]));
				head_out.push(sum);
			}
            x_attn = x_attn.concat(head_out);
		}
        x = linear(x_attn, state_dict['layer'+li+'.attn_wo']);
		x = array_zip_sum(x, x_residual)

        // 2) MLP block
        x_residual = x;
        x = rmsnorm(x);
        x = linear(x, state_dict['layer'+li+'.mlp_fc1']);
		var new_x = [];
		for (var xi of x) new_x.push(xi.relu());
        x = new_x;
        x = linear(x, state_dict['layer'+li+'.mlp_fc2']);
        x = array_zip_sum(x, x_residual);
	}
	var logits = linear(x, state_dict['lm_head']);
    return logits
}

// Let there be Adam, the blessed optimizer and its buffers
var learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;
var m = new Array(params.length); m.fill(0.0); // first moment buffer
var v = new Array(params.length); v.fill(0.0); // second moment buffer

// Repeat in sequence
var num_steps = 1000 // number of training steps
for (var step = 0; step < num_steps; step++) {

    // Take single document, tokenize it, surround it with BOS special token on both sides
    var doc = docs[step % docs.length];
    var tokens = [BOS];
	for (var i = 0; i < doc.length; i++)
		tokens.push(uchars.indexOf(doc[i]));
	tokens.push(BOS);
    var n = Math.min(block_size, tokens.length - 1);

    // Forward the token sequence through the model, building up the computation graph all the way to the loss.
	var keys = new Array(n_layer); keys.fill([]);
	var values = new Array(n_layer); values.fill([]);
    var losses = [];
    for (var pos_id = 0; pos_id < n; pos_id++) {
        var token_id = tokens[pos_id];
		var target_id = tokens[pos_id + 1];
        var logits = gpt(token_id, pos_id, keys, values);
        var probs = softmax(logits);
        var loss_t = probs[target_id].log().mul(-1);
        losses.push(loss_t);
	}
	var losses_sum = new Value(0);
	for(var loss1 of losses) losses_sum = loss1.add(losses_sum);
    var loss = (new Value(1)).truediv(n).mul(losses_sum); // final average loss over the document sequence. May yours be low.

    // Backward the loss, calculating the gradients with respect to all model parameters.
    loss.backward();

    // Adam optimizer update: update the model parameters based on the corresponding gradients.
    var lr_t = learning_rate * (1 - step / num_steps); // linear learning rate decay
    for (var i = 0; i < params.length; i++) {
		var p = params[i];
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
        v[i] = beta2 * v[i] + (1 - beta2) * (p.grad ** 2);
        var m_hat = m[i] / (1 - beta1 ** (step + 1));
        var v_hat = v[i] / (1 - beta2 ** (step + 1));
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam);
        p.grad = 0;
	}

    print("step "+(step+1)+" / "+num_steps+" | loss "+loss.data);
}

// Inference: may the model babble back to us
var temperature = 0.5 // in (0, 1], control the "creativity" of generated text, low to high
print("--- inference (new, hallucinated names) ---");
for (var sample_idx = 0; sample_idx < 20; sample_idx++) {
	var keys = new Array(n_layer); keys.fill([]);
	var values = new Array(n_layer); values.fill([]);
    var token_id = BOS;
    var sample = [];
    for (var pos_id = 0; pos_id < block_size; pos_id++) {
        var logits = gpt(token_id, pos_id, keys, values);
        var probs = [];
		for (var l of logits)
			probs.push(l.truediv(temperature));
		probs = softmax(probs);

		var weights = [];
		for(var p of probs) weights.push(p.data);

        var token_id = random_choices(array_range(vocab_size), weights)[0]
        if (token_id == BOS)
            break
        sample.push(uchars[token_id]);
	}
	print("sample "+(sample_idx+1)+": "+sample.join(''));
}