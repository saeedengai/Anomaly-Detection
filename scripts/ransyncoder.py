class RANSynCoders():
    """ class for building, training, and testing rancoders models """
    def __init__(
            self,
            # Rancoders inputs:
            n_estimators: int = 100,
            max_features: int = 3,
            encoding_depth: int = 2,
            latent_dim: int = 2, 
            decoding_depth: int = 2,
            activation: str = 'linear',
            output_activation: str = 'linear',
            delta: float = 0.05,  # quantile bound for regression
            # Syncrhonization inputs
            synchronize: bool = False,
            force_synchronization: bool = True,  # if synchronization is true but no significant frequencies found
            min_periods: int = 3,  # if synchronize and forced, this is the minimum bound on cycles to look for in train set
            freq_init: Optional[List[float]] = None,  # initial guess for the dominant angular frequency
            max_freqs: int = 1,  # the number of sinusoidal signals to fit
            min_dist: int = 60,  # maximum distance for finding local maximums in the PSD
            trainable_freq: bool = False,  # whether to make the frequency a variable during layer weight training
            bias: bool = True,  # add intercept (vertical displacement)
    ):
        # Rancoders inputs:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.encoding_depth = encoding_depth
        self.latent_dim = latent_dim
        self.decoding_depth = decoding_depth
        self.activation = activation
        self.output_activation = output_activation
        self.delta = delta
        
        # Syncrhonization inputs
        self.synchronize = synchronize
        self.force_synchronization = force_synchronization
        self.min_periods = min_periods
        self.freq_init = freq_init  # in radians (angular frequency)
        self.max_freqs = max_freqs
        self.min_dist = min_dist
        self.trainable_freq = trainable_freq
        self.bias = bias
        
        # set all variables to default to float32
        tf.keras.backend.set_floatx('float32')

    def build(self, n_features: int):
        # ---- set optimizers (since we won't use Model.compile) ----
        self._ranc_opt = tf.keras.optimizers.Adam()
        self._sin_opt  = tf.keras.optimizers.Adam()

        # ---- instantiate models directly (subclass/eager style) ----
        self.rancoders = RANCoders(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            encoding_depth=self.encoding_depth,
            latent_dim=self.latent_dim,
            decoding_depth=self.decoding_depth,
            delta=self.delta,
            activation=self.activation,
            output_activation=self.output_activation,
            name='rancoders'
        )
        # build weights by calling once on a dummy batch
        _ = self.rancoders(tf.zeros((1, n_features), dtype=tf.float32), training=False)

        if self.synchronize:
            self.sincoder = sincoder(freq_init=self.freq_init, trainable_freq=self.trainable_freq)
            _ = self.sincoder(tf.zeros((1, n_features), dtype=tf.float32), training=False)


        
    def fit(
            self, 
            x: np.ndarray, 
            t: np.ndarray,
            epochs: int = 25, 
            batch_size: int = 360, 
            shuffle: bool = True, 
            freq_warmup: int = 10,  # number of warmup epochs to prefit the frequency
            sin_warmup: int = 10,  # number of warmup epochs to prefit the sinusoidal representation
            pos_amp: bool = True,  # whether to constraint amplitudes to be +ve only
    ):
        
        # Prepare the training batches.
        dataset = tf.data.Dataset.from_tensor_slices(
            (x.astype(np.float32), t.astype(np.float32))
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=x.shape[0], reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)  # <-- ALWAYS batch, even if not shuffled



        # --- build models once (no freqcoder; spectral init instead) ---
        if not hasattr(self, "rancoders") or (self.synchronize and not hasattr(self, "sincoder")):
            # If no initial freq_init given, compute from the current x via spectrum
            if self.synchronize and not self.freq_init:
                ang_list = _spectral_freq_init_from_batch(
                    x, self.min_periods, self.max_freqs, self.min_dist, self.force_synchronization
                )
                if ang_list is None:
                    # no oscillation signal at all -> turn off sync
                    self.synchronize = False
                    print("[sync] No common oscillation found; turning synchronization OFF.")
                elif len(ang_list) == 0:
                    # no strong peaks but user requested force
                    if self.force_synchronization:
                        # safe fallback ~1000-sample period
                        self.freq_init = [2.0 * np.pi / 1000.0]
                        print("[sync] No strong peaks; forcing sync with fallback period ~1000 samples.")
                    else:
                        self.synchronize = False
                        print("[sync] No common oscillation; synchronization OFF.")
                else:
                    self.freq_init = ang_list
                    periods = [(2.0 * np.pi) / w for w in ang_list]
                    print(f"[sync] Spectral init periods: {periods}")

            # build rancoders (+ sincoder if synchronize=True)
            self.build(x.shape[1])  # <-- RIGHT

        
        # pretraining step :
        if sin_warmup > 0 and self.synchronize and not getattr(self, "_sine_warmup_done", False):
            for epoch in range(sin_warmup):
                print("\nStart of sine representation pre-train epoch %d" % (epoch,))
                for step, (x_batch, t_batch) in enumerate(dataset):
                    # Train the sine wave encoder
                    with tf.GradientTape() as tape:
                        s = self.sincoder(t_batch, training=True)
                        s_loss = quantile_loss(0.5, x_batch, s)

                    grads = tape.gradient(s_loss, self.sincoder.trainable_weights)
                    self._sin_opt.apply_gradients(zip(grads, self.sincoder.trainable_weights))

                print("sine_loss:", tf.reduce_mean(s_loss).numpy(), end='\r')
            
            # invert params (all amplitudes should either be -ve or +ve). Here we make them +ve
            if pos_amp:
                sinc_layer = _get_sinc_layer_from_model(self.sincoder)

                # Work in eager/NumPy to avoid slice-assign on Variables
                amp_now  = sinc_layer.amp.numpy()      # (F, max_freqs)
                wb_now   = sinc_layer.wb.numpy()       # (F, max_freqs)
                disp_now = sinc_layer.disp.numpy()     # (F,)

                neg_mask = amp_now[:, 0] < 0.0

                a_adj = np.where(neg_mask, -amp_now[:, 0], amp_now[:, 0])
                wb_adj = np.where(neg_mask, wb_now[:, 0] + np.pi, wb_now[:, 0])

                # Robust phase wrap to [0, 2π)
                two_pi = 2.0 * np.pi
                wb_adj = np.mod(wb_adj, two_pi)

                g_adj = np.where(neg_mask, disp_now - a_adj, disp_now)

                _assign_sinc_weights(sinc_layer, a_adj, wb_adj, g_adj)

                self._sine_warmup_done = True
            

        # train anomaly detector
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            if self.synchronize:
                for step, (x_batch, t_batch) in enumerate(dataset):
                    # Train the sine wave encoder
                    with tf.GradientTape() as tape:
                        s = self.sincoder(t_batch, training=True)
                        s_loss = quantile_loss(0.5, x_batch, s)
                    grads = tape.gradient(s_loss, self.sincoder.trainable_weights)
                    self._sin_opt.apply_gradients(zip(grads, self.sincoder.trainable_weights))
                    
                    # synchronize batch (robust layer access + stable trig)
                    sinc_layer = _get_sinc_layer_from_model(self.sincoder)
                    freq = sinc_layer.freq     # (max_freqs,)
                    wb   = sinc_layer.wb       # (F, max_freqs)
                    amp  = sinc_layer.amp      # (F, max_freqs)
                    disp = sinc_layer.disp     # (F,)

                    freq_safe = tf.maximum(freq, 1e-6)
                    b = wb / freq_safe
                    b_sync = b - tf.expand_dims(b[:, 0], axis=-1)

                    th_sync = tf.expand_dims(tf.expand_dims(freq, axis=0), axis=0) * (
                        tf.expand_dims(t_batch, axis=-1) + tf.expand_dims(b_sync, axis=0)
                    )

                    # sin(f*(π/(2f) - b)) == cos(f*b)
                    e = (x_batch - s) * tf.cos(freq[0] * b[:, 0])

                    x_batch_sync = tf.reduce_sum(
                        tf.expand_dims(amp, axis=0) * tf.sin(th_sync), axis=-1
                    ) + disp + e


                    
                    # train the rancoders
                    with tf.GradientTape() as tape:
                        o_hi, o_lo = self.rancoders(x_batch_sync, training=True)
                        target = tf.tile(tf.expand_dims(x_batch_sync, axis=0), (self.n_estimators, 1, 1))
                        o_hi_loss = quantile_loss(1.0 - self.delta, target, o_hi)
                        o_lo_loss = quantile_loss(self.delta,          target, o_lo)
                        o_loss = o_hi_loss + o_lo_loss
                    grads = tape.gradient(o_loss, self.rancoders.trainable_weights)
                    self._ranc_opt.apply_gradients(zip(grads, self.rancoders.trainable_weights))
                print(
                    "sine_loss:", tf.reduce_mean(s_loss).numpy(), 
                    "upper_bound_loss:", tf.reduce_mean(o_hi_loss).numpy(), 
                    "lower_bound_loss:", tf.reduce_mean(o_lo_loss).numpy(), 
                    end='\r'
                )
            else:
                for step, (x_batch, t_batch) in enumerate(dataset):
                    with tf.GradientTape() as tape:
                        o_hi, o_lo = self.rancoders(x_batch, training=True)
                        target = tf.tile(tf.expand_dims(x_batch, axis=0), (self.n_estimators, 1, 1))
                        o_hi_loss = quantile_loss(1.0 - self.delta, target, o_hi)
                        o_lo_loss = quantile_loss(self.delta,          target, o_lo)
                        o_loss = o_hi_loss + o_lo_loss

                    grads = tape.gradient(o_loss, self.rancoders.trainable_weights)
                    self._ranc_opt.apply_gradients(zip(grads, self.rancoders.trainable_weights))

                print(
                    "upper_bound_loss:", tf.reduce_mean(o_hi_loss).numpy(), 
                    "lower_bound_loss:", tf.reduce_mean(o_lo_loss).numpy(), 
                    end='\r'
                )
            
    def predict(self, x: np.ndarray, t: np.ndarray, batch_size: int = 1000, desync: bool = False):
        # Prepare the training batches.
        dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), t.astype(np.float32)))
        dataset = dataset.batch(batch_size)
        batches = int(np.ceil(x.shape[0] / batch_size))
        
        # loop through the batches of the dataset.
        if self.synchronize:
            s, x_sync, o_hi, o_lo = [None] * batches, [None] * batches, [None] * batches, [None] * batches
            for step, (x_batch, t_batch) in enumerate(dataset):
                # forward pass of the sincoder
                s_i = self.sincoder(t_batch).numpy()

                # robustly get the actual sinc layer (works across notebook reloads)
                sinc_layer = _get_sinc_layer_from_model(self.sincoder)
                freq = sinc_layer.freq   # shape: (max_freqs,)
                wb   = sinc_layer.wb     # shape: (F, max_freqs)
                amp  = sinc_layer.amp    # shape: (F, max_freqs)
                disp = sinc_layer.disp   # shape: (F,)

                # Safe divide & stable trig
                freq_safe = tf.maximum(freq, 1e-6)       # avoid div-by-zero
                b = wb / freq_safe                       # phase shift(s), shape (F, max_freqs)
                b_sync = b - tf.expand_dims(b[:, 0], axis=-1)

                # synchronized angle for aligned signal
                th_sync = tf.expand_dims(tf.expand_dims(freq, axis=0), axis=0) * (
                    tf.expand_dims(t_batch, axis=-1) + tf.expand_dims(b_sync, axis=0)
                )

                # sin(f*(π/(2f) - b)) == cos(f*b)  -> avoids inf/NaN when f≈0
                e = (x_batch - s_i) * tf.cos(freq[0] * b[:, 0])

                # final synchronized batch
                x_sync_i = (tf.reduce_sum(tf.expand_dims(amp, axis=0) * tf.sin(th_sync), axis=-1)
                            + disp + e).numpy()
  
                o_hi_i, o_lo_i = self.rancoders(x_sync_i)
                o_hi_i, o_lo_i = tf.transpose(o_hi_i, [1,0,2]).numpy(), tf.transpose(o_lo_i, [1,0,2]).numpy()
                if desync:
                    o_hi_i, o_lo_i = self.predict_desynchronize(x_batch, x_sync_i, o_hi_i, o_lo_i)
                s[step], x_sync[step], o_hi[step], o_lo[step]  = s_i, x_sync_i, o_hi_i, o_lo_i
            return (
                np.concatenate(s, axis=0), 
                np.concatenate(x_sync, axis=0), 
                np.concatenate(o_hi, axis=0), 
                np.concatenate(o_lo, axis=0)
            )
        else:
            o_hi, o_lo = [None] * batches, [None] * batches
            for step, (x_batch, t_batch) in enumerate(dataset):
                o_hi_i, o_lo_i = self.rancoders(x_batch)
                o_hi_i, o_lo_i = tf.transpose(o_hi_i, [1,0,2]).numpy(), tf.transpose(o_lo_i, [1,0,2]).numpy()
                o_hi[step], o_lo[step]  = o_hi_i, o_lo_i
            return np.concatenate(o_hi, axis=0), np.concatenate(o_lo, axis=0)
        
    def save(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = os.path.join(os.getcwd(), "ransyncoders.z")
        file = {'params': self.get_config()}
        if hasattr(self, "freqcoder"):  # guard
            file['freqcoder'] = {'model': self.freqcoder.to_json(), 'weights': self.freqcoder.get_weights()}
        if self.synchronize:
            file['sincoder'] = {'model': self.sincoder.to_json(), 'weights': self.sincoder.get_weights()}
        file['rancoders'] = {'model': self.rancoders.to_json(), 'weights': self.rancoders.get_weights()}
        dump(file, filepath, compress=True)

    @classmethod
    def load(cls, filepath: Optional[str] = None):
        if filepath is None:
            filepath = os.path.join(os.getcwd(), "ransyncoders.z")
        file = load(filepath)
        inst = cls()
        for param, val in file['params'].items():
            setattr(inst, param, val)
        if 'freqcoder' in file:
            inst.freqcoder = model_from_json(file['freqcoder']['model'], custom_objects={'freqcoder': freqcoder})
            inst.freqcoder.set_weights(file['freqcoder']['weights'])
        if inst.synchronize and 'sincoder' in file:
            inst.sincoder = model_from_json(file['sincoder']['model'], custom_objects={'sincoder': sincoder})
            inst.sincoder.set_weights(file['sincoder']['weights'])
        inst.rancoders = model_from_json(file['rancoders']['model'], custom_objects={'RANCoders': RANCoders})
        inst.rancoders.set_weights(file['rancoders']['weights'])
        return inst


    
    def predict_desynchronize(self, x: np.ndarray, x_sync: np.ndarray, o_hi: np.ndarray, o_lo: np.ndarray):
        if self.synchronize:
            E = (o_hi + o_lo)/ 2  # expected values
            deviation = tf.expand_dims(x_sync, axis=1) - E  # input (synchronzied) deviation from expected
            deviation = self.desynchronize(deviation)  # desynchronize
            E = tf.expand_dims(x, axis=1) - deviation  # expected values in desynchronized form
            offset = (o_hi - o_lo) / 2  # this is the offet from the expected value
            offset = abs(self.desynchronize(offset))  # desynch
            o_hi, o_lo = E + offset, E - offset  # add bound displacement to expected values
            return o_hi.numpy(), o_lo.numpy()  
        else:
            raise ParameterError('synchronize', 'parameter not set correctly for this method')
    
    def desynchronize(self, e: np.ndarray):
        if not self.synchronize:
            raise ParameterError('synchronize', 'parameter not set correctly for this method')

        sinc_layer = _get_sinc_layer_from_model(self.sincoder)
        freq = sinc_layer.freq
        wb   = sinc_layer.wb

        freq_safe = tf.maximum(freq, 1e-6)
        b = wb / freq_safe  # (F, max_freqs)

        # sin(f*(π/(2f)+b)) == cos(f*b)
        return (e * tf.cos(freq[0] * b[:, 0])).numpy()


        
        
    def get_config(self):
        config = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "encoding_depth": self.encoding_depth,
            "latent_dim": self.latent_dim,
            "decoding_depth": self.decoding_depth,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "delta": self.delta,
            "synchronize": self.synchronize,
            "force_synchronization": self.force_synchronization,
            "min_periods": self.min_periods,
            "freq_init": self.freq_init,
            "max_freqs": self.max_freqs,
            "min_dist": self.min_dist,
            "trainable_freq": self.trainable_freq,
            "bias": self.bias,
        }
        return config
        
        
# Loss function
def quantile_loss(q, y, f):
    e = (y - f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)


class ParameterError(Exception):

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
