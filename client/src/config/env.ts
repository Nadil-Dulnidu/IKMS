const BACKEND_URL = `${process.env.NEXT_PUBLIC_BACKEND_URL}`;

if (!BACKEND_URL) {
  throw new Error("Add your Backend URL to the .env file");
}

export { BACKEND_URL};
