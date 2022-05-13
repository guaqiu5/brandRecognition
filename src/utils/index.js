import { getImg } from "./getImg.js";
import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

const NUM_CLASSES=3


window.onload = async () =>{
    const {imgs,mapLabels} =await getImg();
    console.log(imgs,mapLabels)
    const surface=tfvis.visor().surface({name:'输入示例'})
    imgs.forEach(img=>{
        surface.drawArea.appendChild(img)
    })
    //加载模型
    const mobilenet=await tf.loadLayersModel('./model/model.json')
    mobilenet.summary();
    //截断
    const layer =mobilenet.getLayer('conv_pw_13_relu')
    const lessMobilenet=tf.model({
        inputs:mobilenet.inputs,
        outputs:layer.output
    })

    //双层神经网络
    const model=tf.sequential()
    //高维度卷积flat变为1维
    model.add(tf.layers.flatten({
        //outputShape第一个为空(保留位，确定输入的个数），所以要slice
        inputShape:layer.outputShape.slice(1)
    }))
    model.add(tf.layers.dense({
        units:10,
        activation:'relu'
    }))
    model.add(tf.layers.dense({
        units:NUM_CLASSES,
        activation:'softmax'
    }))

    //设置损失函数和 优化器
    model.compile({loss:'categoricalCrossentropy',optimizer:tf.train.adam()})

    //输入数据喂给截断模型
    //截断模型的input就是mobilenet的input 
    //要变成tensor后再归一化
}