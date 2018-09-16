/*
	zuki_nn combines RSI Bull and Bear
	Uses zuki_nn to predict price, and combine with RSI analysis from RSI_BULL_BEAR.
	Use different RSI-strategies depending on a longer trend
	16 sep 2018

	Preferences:
	zuki_nn: https://github.com/gekkowarez/gekko-neuralnet
	RSI_BULL_BEAR: (CC-BY-SA 4.0) Tommie Hansen :
	https://github.com/xFFFFF/Gekko-Strategies/tree/master/RSI_BULL_BEAR
	
	(CC-BY-SA 4.0) Son Nguyen
	https://creativecommons.org/licenses/by-sa/4.0/
	
*/

var convnetjs = require('convnetjs');
var math = require('mathjs');
var fs = require('fs');
var log = require('../core/log.js');
var config = require ('../core/util.js').getConfig();
var SMMA = require('./indicators/SMMA.js');


// var nndatafile = config.filewriter.nnfilepath + 'trained.js'; // the nn state needs to be saved between runs this is the location for it
var nndatafile = 'nn_files/trained.js';

var rsiBullBear = {
		/* INIT */
	init: function(settingParams, strategy)
	{
		this.settings = settingParams;
		this.strategy = strategy;
		this.name = 'RSI Bull and Bear';
		this.requiredHistory = config.tradingAdvisor.historySize;
		this.resetTrend();		
		
		// debug? set to flase to disable all logging/messages (improves performance)
		this.debug = false;
		
		// performance
		config.backtest.batchSize = 1000; // increase performance
		config.silent = true;
		config.debug = false;
		
		// add indicators
		this.strategy.addIndicator('maSlow', 'SMA', this.settings.SMA_long );
		this.strategy.addIndicator('maFast', 'SMA', this.settings.SMA_short );
		this.strategy.addIndicator('BULL_RSI', 'RSI', { interval: this.settings.BULL_RSI });
		this.strategy.addIndicator('BEAR_RSI', 'RSI', { interval: this.settings.BEAR_RSI });
		
		// debug stuff
		this.startTime = new Date();
		this.stat = {
			bear: { min: 100, max: 0 },
			bull: { min: 100, max: 0 }
		};
		
	}, // init()
	
	
	/* RESET TREND */
	resetTrend: function()
	{
		var trend = {
			duration: 0,
			direction: 'none',
			longPos: false,
		};
	
		this.trend = trend;
	},
	
	/* get lowest/highest for backtest-period */
	lowHigh: function( rsi, type )
	{
		let cur;
		if( type == 'bear' ) {
			cur = this.stat.bear;
			if( rsi < cur.min ) this.stat.bear.min = rsi; // set new
			if( rsi > cur.max ) this.stat.bear.max = rsi;
		}
		else {
			cur = this.stat.bull;
			if( rsi < cur.min ) this.stat.bull.min = rsi; // set new
			if( rsi > cur.max ) this.stat.bull.max = rsi;
		}
	},
	
	
	/* CHECK */
	check: function()
	{
		
		// get all indicators
		let ind = this.strategy.indicators,
			maSlow = ind.maSlow.result,
			maFast = ind.maFast.result,
			rsi;
			
		// BEAR TREND
		if( maFast < maSlow )
		{
			rsi = ind.BEAR_RSI.result;
			if( rsi > this.settings.BEAR_RSI_high ) return this.short();
			else if( rsi < this.settings.BEAR_RSI_low ) return this.long();
			
			if(this.debug) this.lowHigh( rsi, 'bear' );
			//log.debug('BEAR-trend');
		}

		// BULL TREND
		else
		{
			rsi = ind.BULL_RSI.result;
			if( rsi > this.settings.BULL_RSI_high ) return this.short();
			else if( rsi < this.settings.BULL_RSI_low ) return this.long();
			if(this.debug) this.lowHigh( rsi, 'bull' );
			//log.debug('BULL-trend');
		}
	
	}, // check()
	
	
	/* LONG */
	long: function()
	{
		if( this.trend.direction !== 'up' ) // new trend? (only act on new trends)
		{
			this.resetTrend();
			this.trend.direction = 'up';
			return 'long';
			//log.debug('go long');
		}
		
		if(this.debug)
		{
			this.trend.duration++;
			log.debug ('Long since', this.trend.duration, 'candle(s)');
		}
		return 'stay';
	},
	
	
	/* SHORT */
	short: function()
	{
		// new trend? (else do things)
		if( this.trend.direction !== 'down' )
		{
			this.resetTrend();
			this.trend.direction = 'down';
			return 'short';
		}
		
		if(this.debug)
		{
			this.trend.duration++;
			log.debug ('Short since', this.trend.duration, 'candle(s)');
		}
		return 'stay';
	},
	
	
	/* END backtest */
	end: function(){
		
		let seconds = ((new Date()- this.startTime)/1000),
			minutes = seconds/60,
			str;
			
		minutes < 1 ? str = seconds + ' seconds' : str = minutes + ' minutes';
		
		log.debug('====================================');
		log.debug('Finished in ' + str);
		log.debug('====================================');
		
		if(this.debug)
		{
			let stat = this.stat;
			log.debug('RSI low/high for period');
			log.debug('BEAR low/high: ' + stat.bear.min + ' / ' + stat.bear.max);
			log.debug('BULL low/high: ' + stat.bull.min + ' / ' + stat.bull.max);
		}
	}
};

var zuki_nn = {
  // stores the candles
  priceBuffer : [],
  predictionCount : 0,

  batchsize : 1,
  // no of neurons for the layer
  layer_neurons : 0,
  // activaction function for the first layer, when neurons are > 0
  layer_activation : 'tanh',
  // normalization factor
  scale : 1,
  // stores the last action (buy or sell)
  prevAction : 'wait',
  //stores the price of the last trade (buy/sell)
  prevPrice : 0,
  // counts the number of triggered stoploss events
  stoplossCounter : 0,

  // if you want the bot to hodl instead of selling during a small dip
  // use the hodle_threshold. e.g. 0.95 means the bot won't sell
  // unless the price drops 5% below the last buy price (this.privPrice)
  hodle_threshold : 1,

  // init the strategy
  init : function(settingParams, strategy) {
  	this.settings = settingParams;
  	this.strategy = strategy;
    this.name = 'NN RSI BULL BEAR';
    this.requiredHistory = config.tradingAdvisor.historySize;

    // smooth the input to reduce the noise of the incoming data
    this.SMMA = new SMMA(5);

	//always create a new instance of convnetjs network
	this.nn = new convnetjs.Net();	
	
	if (fs.existsSync(nndatafile)){
	 this.nn.fromJSON(JSON.parse(fs.readFileSync(nndatafile,'utf8')));	
	}else{

		let layers = [
		  {type:'input', out_sx:1, out_sy:1, out_depth: 1},
		  {type:'fc', num_neurons: this.layer_neurons, activation: this.layer_activation},
		  {type:'regression', num_neurons: 1}
		];

		this.nn.makeLayers( layers );
	}
	
	this.trainer = new convnetjs.SGDTrainer(this.nn, {
	  learning_rate: this.settings.learning_rate,
	  momentum: this.settings.momentum,
	  batch_size: this.batchsize,
	  l2_decay: this.settings.decay
	});		


    this.strategy.addIndicator('stoploss', 'StopLoss', {
      threshold : this.settings.stoploss_threshold
    });

    this.hodle_threshold = this.settings.hodle_threshold || 1;
  },

  learn : function () {
    for (let i = 0; i < this.priceBuffer.length - 1; i++) {
      let data = [this.priceBuffer[i]];
      let current_price = [this.priceBuffer[i + 1]];
      let vol = new convnetjs.Vol(data);
      this.trainer.train(vol, current_price);
       this.predictionCount++;
    }
  },

  setNormalizeFactor : function(candle) {
    this.scale = Math.pow(10,Math.trunc(candle.high).toString().length+2);
    log.debug('Set normalization factor to',this.scale);
  },

  update : function(candle)
  {
    // play with the candle values to finetune this
    this.SMMA.update( (candle.high + candle.close + candle.low + candle.vwp) /4);
    let smmaFast = this.SMMA.result;

    if (1 === this.scale && 1 < candle.high && 0 === this.predictionCount) this.setNormalizeFactor(candle);

    this.priceBuffer.push(smmaFast / this.scale );
    if (2 > this.priceBuffer.length) return;

     for (i=0;i<3;++i)
      this.learn();

    while (this.settings.price_buffer_len < this.priceBuffer.length) this.priceBuffer.shift();
  },

  onTrade: function(event) {

    if ('buy' === event.action) {
      this.strategy.indicators.stoploss.long(event.price);
    }
    // store the previous action (buy/sell)
    this.prevAction = event.action;
    // store the price of the previous trade
    this.prevPrice = event.price;

  },

  predictCandle : function() {
    let vol = new convnetjs.Vol(this.priceBuffer);
    let prediction = this.nn.forward(vol);
    return prediction.w[0];
  },

  check : function(candle) {
    if(this.predictionCount > this.settings.min_predictions)
    {
      if (
          'buy' === this.prevAction
          && this.settings.stoploss_enabled
          && 'stoploss' === this.strategy.indicators.stoploss.action
      ) {
        this.stoplossCounter++;
        log.debug('>>>>>>>>>> STOPLOSS triggered <<<<<<<<<<');
        return 'short';
      }

      let prediction = this.predictCandle() * this.scale;
      let currentPrice = candle.close;
      let meanp = math.mean(prediction, currentPrice);
      let meanAlpha = (meanp - currentPrice) / currentPrice * 100;


      // sell only if the price is higher than the buying price or if the price drops below the threshold
      // a hodle_threshold of 1 will always sell when the NN predicts a drop of the price. play with it!
      let signalSell = candle.close > this.prevPrice || candle.close < (this.prevPrice*this.hodle_threshold);

      let signal = meanp < currentPrice;
      if ('buy' !== this.prevAction && signal === false  && meanAlpha> this.settings.threshold_buy )
      {

        log.debug("Buy - Predicted variation: ",meanAlpha);
        return 'long';
      }
      else if
      ('sell' !== this.prevAction && signal === true && meanAlpha < this.settings.threshold_sell && signalSell)
      {

        log.debug("Sell - Predicted variation: ",meanAlpha);
        return 'short';

      }

    }
    return 'stay';
  },

  end : function() {
	log.debug("NN output to store: ", JSON.stringify(this.nn.toJSON()));
	var fileoutput = JSON.stringify(this.nn.toJSON());
    fs.writeFileSync(nndatafile, fileoutput, function (err) {
      if (err) throw err;
        console.log('Learn state saved!');
    });	  
    log.debug('Triggered stoploss',this.stoplossCounter,'times');
  }
};

var strategy = {
  // init the strategy
  init : function()
  {
  	rsiBullBear.init(this.settings, this);
  	zuki_nn.init(this.settings, this);
  },

  update : function(candle)
  {
  	zuki_nn.update(candle);
  },

  check : function(candle)
  {
  	var resultRSIBullBear = rsiBullBear.check(candle);
  	var resultZuki_nn = zuki_nn.check(candle);
  	if ((resultRSIBullBear === 'long' && resultZuki_nn !== 'short')
  		|| (resultZuki_nn === 'long' && resultRSIBullBear !== 'short')) {
  		this.advice('long');
  	} else if ((resultRSIBullBear === 'short' && resultZuki_nn !== 'long')
  		|| (resultZuki_nn === 'short' && resultRSIBullBear !== 'long')) {
  		this.advice('short');
  	}
  },

  end : function()
  {
  	rsiBullBear.end();
  	zuki_nn.end();
  }
};

module.exports = strategy;