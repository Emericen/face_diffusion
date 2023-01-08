Diffusion Models for content generation are exciting as hell. In this project, I created a fake face generation model that one could practically train on a personal computer. Just as most of us entered the realm of deep learning with the MNIST project, for me this project serves as a starting point for content generation models. 

I'm running RTX 3060 on my PC, and I scraped 30,000 fake face images from <a href="https://thispersondoesnotexist.com/">thispersondoesnotexist</a> to train my model. Here's what they look like

<img src="https://github.com/Emericen/face_diffusion/blob/master/assets/demo-1.png?raw=true" alt="alt text z" style="zoom:50%;" />

The gist of diffusion models is essentially forward and backward diffusion processes. We turn images into pure noise by forward diffusion, and use a backward diffusion process to turn random noise into images. Our neural network is used in the backward diffusion process. Instead of learning about images like in GANs, our model learn about noise distribution and how to clear out said noise.

We start with the forward diffusion process, which gradually adds random noise to an image, making it a complete image of noise by the end of certain amount of timestep. This is a Markov process, where the state of the current image depends only on the previous image. We denote the process as q.

Let's denote x<sub>0</sub> = original image, T = total time step. And x<sub>1</sub> to x<sub>T</sub> are more and more noisy version of x<sub>0</sub>. The way noise is sampled at time t is described by 

$$
\begin{align*}
q(x_t|x_{t-1}) &=N(x_t;\sqrt{1-{\beta}_t}x_{t-1}, {{\beta}_t}I) \\
&=\sqrt{1-{{\beta}_t}}x_{t-1}+\sqrt{{\beta}_t}\epsilon
\end{align*}
$$

$$
{\text{Where }} x_t=\text{output}, \sqrt{1-{{\beta}_t}}=\text{mean}, {{\beta}_t}I =\text{fixed variance}, {\epsilon\sim}N(0,1) \text{ meaning avg 0 and std 1}
$$

Variance at time t is basically how much noise we'd like to generate. Further, we define the following

$$
\begin{align*}
&{\alpha}_t=1-{\beta}_t \\
&{\overline{{\alpha}}_t} = \prod_{i=1}^{t} {\alpha}_t
\end{align*}
$$

Combining this into our definition of q, we have

$$
\begin{align*}
q(x_t|x_{t-1})&=\sqrt{1-{{\beta}_t}}x_{t-1}+\sqrt{{\beta}_t}\epsilon \\
&=\sqrt{{\alpha}_t}x_{t-1}+\sqrt{1-{\alpha}_t}\epsilon \\
&=\sqrt{{\alpha}_t{\alpha}_{t-1}}x_{t-2}+\sqrt{1-{\alpha}_t{\alpha}_{t-1}}\epsilon \\
&=\sqrt{{\alpha}_t{\alpha}_{t-1}{\alpha}_{t-2}}x_{t-3}+\sqrt{1-{\alpha}_t{\alpha}_{t-1}{\alpha}_{t-2}}\epsilon \\
&=\sqrt{\overline{{\alpha}}_t}x_0+\sqrt{1-\overline{{\alpha}}_t}\epsilon \\
\end{align*}
$$

This formula gives us the ability to calculate the noised image at any time step, which allows us to not have to loop over each timestep a.k.a O(N) to O(1). Here's what forward diffusion process looks like on an actual image

<img src="https://github.com/Emericen/face_diffusion/blob/master/assets/demo-2.png?raw=true" alt="alt text z" style="zoom:50%;" />

I trained for 50 epochs on the previously mentioned 30,000 face images, which took about 6 to 7 hours. I also sampled the image generation process every 5 epochs to observe the training progression. Here's what it looks like

<img src="https://github.com/Emericen/face_diffusion/blob/master/assets/demo-3.jpg?raw=true" alt="alt text z" style="zoom:50%;" />

And here are some images it can generate after the 50-epoch training

<img src="https://github.com/Emericen/face_diffusion/blob/master/assets/demo-4.jpg?raw=true" alt="alt text z" style="zoom:50%;" />

Again, keep in mind that this is only a starting point for these types of generative diffusion models. There are many improvements to the current version; some of the ones I can quickly think of include
<ul>
<li>More epochs in larger batches with more powerful hardware</li>
<li>Using bigger U-Nets</li>
<li>Adding Residual Blocks</li>
<li>Incorporating Attention Unit</li>
<li>Adding other conditionals such as text descriptions</li>
</ul>

That being said, considering the small amount of effort I made in creating this model versus what it can do, I think we've seen the efficacy of diffusion models.