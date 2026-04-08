const crypto = require('crypto');
const { createClient } = require('@supabase/supabase-js');

const ACTIVE_EVENTS = new Set(['subscription.active', 'subscription.created']);

// Fix 5: module-level Supabase client
const sbAdmin = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

function verifySignature(rawBody, headers, secret) {
  const msgId = headers['webhook-id'];
  const msgTimestamp = headers['webhook-timestamp'];
  const msgSignature = headers['webhook-signature'];
  if (!msgId || !msgTimestamp || !msgSignature) return false;

  // Fix 2: timestamp replay-window check
  const now = Math.floor(Date.now() / 1000);
  const ts  = parseInt(msgTimestamp, 10);
  if (isNaN(ts) || Math.abs(now - ts) > 300) return false;

  const toSign = `${msgId}.${msgTimestamp}.${rawBody}`;
  const secretBytes = Buffer.from(secret.replace(/^whsec_/, ''), 'base64');
  const computed = crypto
    .createHmac('sha256', secretBytes)
    .update(toSign)
    .digest('base64');

  return msgSignature.split(' ').some(sig => {
    const [version, val] = sig.split(',');
    // Fix 1: timing-safe comparison
    const a = Buffer.from(val, 'base64');
    const b = Buffer.from(computed, 'base64');
    return version === 'v1' && a.length === b.length && crypto.timingSafeEqual(a, b);
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

  // Fix 3: wrap getRawBody in try/catch
  let rawBody;
  try {
    rawBody = await getRawBody(req);
  } catch (e) {
    return res.status(400).json({ error: 'Failed to read request body' });
  }

  const isValid = verifySignature(rawBody, req.headers, process.env.POLAR_WEBHOOK_SECRET);
  if (!isValid) return res.status(401).json({ error: 'Invalid signature' });

  // Fix 4: wrap JSON.parse in try/catch
  let event;
  try {
    event = JSON.parse(rawBody);
  } catch (e) {
    return res.status(400).json({ error: 'Invalid JSON' });
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
};
