"use client";

import { useEffect, useRef } from "react";

// Sample a torus knot and construct a tube in its local normal plane.
function center(t: number) {
  const radius = 1.32 + .52 * Math.cos(3*t);
  return [radius*Math.cos(2*t), radius*Math.sin(2*t), .78*Math.sin(3*t)];
}
const rings = 144, sides = 16;
const vertices = Array.from({ length: rings }, (_, i) => {
  const t = i * 2*Math.PI/rings, p = center(t), following = center(t+.001);
  const tangent = following.map((v,k) => v-p[k]);
  const length = Math.hypot(...tangent);
  const [x,y,z] = tangent.map(v => v/length);
  const h = Math.hypot(x,y), nx = -y/h, ny = x/h;
  return Array.from({length:sides}, (_,j) => {
    const a = j*2*Math.PI/sides, c=.31*Math.cos(a), s=.31*Math.sin(a);
    return [p[0]+nx*c-z*ny*s, p[1]+ny*c+z*nx*s, p[2]+h*s];
  });
});

export default function Sculpture({paused}: {paused:boolean}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const preference = matchMedia("(prefers-reduced-motion: reduce)");
    let frame = 0, visible = true, phase = 0, last = 0;
    function draw(time:number) {
      if (!canvas || !ctx) return;
      const w=canvas.clientWidth, h=canvas.clientHeight, ratio=Math.min(devicePixelRatio,2);
      if (canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)) {
        canvas.width=Math.round(w*ratio); canvas.height=Math.round(h*ratio);
      }
      ctx.setTransform(ratio,0,0,ratio,0,0); ctx.clearRect(0,0,w,h);
      if (!paused && !preference.matches) phase+=Math.min(time-last,40)*.00018;
      last=time;
      const a=.52+phase, b=.7, scale=Math.min(w,h)*.215;
      const points=vertices.map(ring=>ring.map(([x,y,z])=>{
        const u=x*Math.cos(a)-z*Math.sin(a), v=x*Math.sin(a)+z*Math.cos(a);
        return [w/2+u*scale,h/2+(y*Math.cos(b)-v*Math.sin(b))*scale,(v*Math.cos(b)+y*Math.sin(b))];
      }));
      const faces=points.flatMap((ring,i)=>ring.map((p,j)=>{
        const next=points[(i+1)%rings], k=(j+1)%sides;
        const quad=[p,next[j],next[k],ring[k]];
        return {quad,depth:quad.reduce((sum,v)=>sum+v[2],0)/4,j,i};
      })).sort((l,r)=>l.depth-r.depth);
      for(const {quad,depth,j,i} of faces){
        const shine=(1+Math.cos(j*2*Math.PI/sides+.65*Math.sin(i/rings*4*Math.PI+a)))/2;
        const light=21+shine*42+depth*7;
        ctx.fillStyle="hsl("+(248+25*shine)+" 83% "+light+"%)";
        ctx.strokeStyle="hsla(255,100%,87%,.52)"; ctx.lineWidth=.58;
        ctx.beginPath(); quad.forEach(([x,y],k)=>k?ctx.lineTo(x,y):ctx.moveTo(x,y)); ctx.closePath();ctx.fill();ctx.stroke();
      }
      if(visible && !paused && !preference.matches) frame=requestAnimationFrame(draw);
    }
    function refresh(){cancelAnimationFrame(frame);if(visible) frame=requestAnimationFrame(draw);}
    const resize=new ResizeObserver(refresh);resize.observe(canvas);
    const observer=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;refresh();});observer.observe(canvas);
    preference.addEventListener("change",refresh);refresh();
    return ()=>{cancelAnimationFrame(frame);resize.disconnect();observer.disconnect();preference.removeEventListener("change",refresh);};
  },[paused]);
  return <canvas ref={ref} aria-label="Rotating violet wire mesh sculpture" role="img" />;
}
