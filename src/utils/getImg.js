const loadImg = (src) =>{
    return new Promise(resolve =>{
        const img =new Image();
        img.crossOrigin='anonymous';
        img.src=src;
        img.width=224;
        img.height=224;
        img.onload =()=>resolve(img)
    })
}



export const getImg = async () => {
    const taskQueue=[]
    const mapLabels=[]
    for(let i=0;i<30;i++){
        ['android','apple','windows'].forEach((label)=>{
            taskQueue.push(loadImg(`./asset/${label}-${i}.jpg`))
           
            mapLabels.push([
                label==='andorid'? 1 : 0,
                label==='apple' ? 1 : 0,
                label==='windows' ? 1 :0,
            ])
        })
    }

   const imgs=await Promise.all(taskQueue)

    return {
        imgs,
        mapLabels,
    }
}