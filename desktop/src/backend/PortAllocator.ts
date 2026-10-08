import net from 'node:net'

export function allocateLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') return server.close(() => reject(new Error('Unable to allocate a loopback port')))
      const port = address.port
      server.close((error) => error ? reject(error) : resolve(port))
    })
  })
}
