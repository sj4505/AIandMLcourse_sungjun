const crypto = require('crypto');
const { createClient } = require('@supabase/supabase-js');

const ACTIVE_EVENTS = new Set(['subscription.active', 'subscription.created']);

function verifySignature(rawBody, headers, secret) {
  const msgId = headers['webhook-id'];
  const msgTimestamp = headers['webhook-timestamp'];
  const msgSignature = headers['webhook-signature'];
  if (!msgId || !msgTimestamp || !msgSignature) return false;

  const toSign = `${msgId}.${msgTimestamp}.${rawBody}`;
  const secretBytes = Buffer.from(secret.replace(/^whsec_/, ''), 'base64');
  const computed = crypto
    .createHmac('sha256', secretBytes)
    .update(toSign)
    .digest('base64');

  return msgSignature.split(' ').some(sig => {
    const [version, val] = sig.split(',');
    return version === 'v1' && val === computed;
  });
}

function getRawBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => { data += chunk; });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).end();

  const rawBody = await getRawBody(req);
  const isValid = verifySignature(rawBody, req.headers, process.env.POLAR_WEBHOOK_SECRET);
  if (!isValid) return res.status(401).json({ error: 'Invalid signature' });

  const event = JSON.parse(rawBody);
  const email = event.data?.customer?.email;
  if (!email) return res.status(200).json({ ok: true });

  const status = ACTIVE_EVENTS.has(event.type) ? 'active' : 'inactive';
  const subscriptionId = event.data?.id ?? null;

  const sbAdmin = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
  );

  const { error } = await sbAdmin
    .from('users')
    .update({ subscription_status: status, subscription_id: subscriptionId })
    .eq('email', email);

  if (error) return res.status(500).json({ error: error.message });
  return res.status(200).json({ ok: true });
};
