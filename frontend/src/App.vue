<template>
  <div class="min-h-screen bg-gray-100 text-gray-800 p-6">
    <h1 class="text-3xl font-bold mb-4">Facial Recognition: Ethics and Legislation</h1>
    <section class="mb-6">
      <h2 class="text-2xl font-semibold">Ethical Issues</h2>
      <ul class="list-disc list-inside mt-2">
        <li>Bias and inaccuracy in facial recognition for people of color and women</li>
        <li>Violation of individual privacy and consent</li>
        <li>Mass surveillance and chilling effects on freedom of expression</li>
        <li>Misuse by law enforcement without proper oversight</li>
      </ul>
    </section>

    <section class="mb-6">
      <h2 class="text-2xl font-semibold">Legislative Recommendations for Minnesota</h2>
      <ul class="list-disc list-inside mt-2">
        <li>Require warrants for any law enforcement use of facial recognition</li>
        <li>Mandate transparency in how data is collected, stored, and used</li>
        <li>Prohibit facial recognition in schools and public housing</li>
        <li>Establish independent oversight boards</li>
      </ul>
    </section>

    <section class="mb-6">
      <h2 class="text-2xl font-semibold">Try It Yourself</h2>
      <p class="mb-2">Upload an image to run a basic facial recognition model (local simulation).</p>
      <input type="file" @change="handleImageUpload" accept="image/*" class="mb-4" />
      <div v-if="result">
        <h3 class="text-lg font-semibold">Result:</h3>
        <pre class="bg-white p-2 rounded shadow">{{ result }}</pre>
      </div>
    </section>
  </div>
</template>

<script>
export default {
  data() {
    return {
      result: null,
    };
  },
  methods: {
    async handleImageUpload(event) {
      const file = event.target.files[0];
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('https://your-flask-api.onrender.com/api/facial-recognition', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      this.result = data;
    },
  },
};
</script>

<style>
body {
  font-family: 'Arial', sans-serif;
}
</style>
