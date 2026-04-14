const { validateEvent } = require('@polar-sh/sdk/webhooks');
const { createClient } = require('@supabase/supabase-js');

const ACTIVE_EVENTS = new Set(['subscription.active', 'subscription.created']);

const sbAdmin = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

function getRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', chunk => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).end();

  let rawBody;
  try {
    rawBody = await getRawBody(req);
  } catch (e) {
    return res.status(400).json({ error: 'Failed to read request body' });
  }

  let event;
  try {
    event = validateEvent(rawBody, req.headers, process.env.POLAR_WEBHOOK_SECRET);
  } catch (e) {
    return res.status(401).json({ error: 'Invalid signature', detail: e.message });
  }

  const email = event.data?.customer?.email;
  if (!email) return res.status(200).json({ ok: true });

  const status = ACTIVE_EVENTS.has(event.type) ? 'active' : 'inactive';
  const subscriptionId = event.data?.id ?? null;

  const { error } = await sbAdmin
    .from('users')
    .update({ subscription_status: status, subscription_id: subscriptionId })
    .eq('email', email);

  if (error) return res.status(500).json({ error: error.message });
  return res.status(200).json({ ok: true });
}

handler.config = { api: { bodyParser: false } };
module.exports = handler;
